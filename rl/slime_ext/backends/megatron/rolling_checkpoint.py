"""Opt-in rolling native checkpoints; cadence/retention come from the launcher.

Megatron / nvidia-resiliency-ext owns tensor staging, distributed writes and
writer error propagation. Slime saves its dataset cursor first; Megatron rank
zero advances the atomic iteration marker after the model/optimizer/RNG writer
succeeds. Only points created by this policy are eligible for deletion;
historical checkpoints remain untouched.
"""

import inspect
import json
import logging
import os
import shutil
import tempfile
import threading
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)
_OWNER = "slime_rolling_checkpoint_v1"
_COMPLETE = "_SLIME_COMPLETE.json"
_TRACKER = "latest_checkpointed_iteration.txt"
_active_save = ContextVar("slime_rolling_checkpoint_save", default=None)
_applied = False

_NATIVE_DATA_SOURCE_PATHS = (
    "slime.data.data_source.RolloutDataSourceWithBuffer",
    "slime.rollout.data_source.RolloutDataSourceWithBuffer",
)


def enabled(args):
    return bool(getattr(args, "rolling_checkpoint_retain_interval", 0))


def validate_args(args):
    if not enabled(args):
        return
    interval = args.rolling_checkpoint_retain_interval
    if interval < 1 or not args.save_interval or args.save_interval < 1 or interval % args.save_interval:
        raise ValueError("Rolling checkpoint retention must be a positive multiple of --save-interval")
    if not args.save or not args.async_save or args.ckpt_format != "torch_dist":
        raise ValueError("Rolling checkpoints require --save, --async-save and --ckpt-format torch_dist")
    if not getattr(args, "use_persistent_ckpt_worker", False):
        raise ValueError("Rolling checkpoints require --use-persistent-ckpt-worker")
    if args.no_save_optim or args.no_save_rng:
        raise ValueError("Rolling recovery checkpoints require optimizer and RNG state")
    if any(getattr(args, key, False) for key in ("offload_train", "release_train", "use_critic")):
        raise ValueError("Rolling checkpoints currently support the resident Megatron actor without a critic")
    if getattr(args, "save_hf", None) or getattr(args, "save_retain_interval", None):
        raise ValueError("Rolling checkpoints cannot combine with HF saving or native retention")
    data_source_path = getattr(args, "data_source_path", None)
    if data_source_path is not None and data_source_path not in _NATIVE_DATA_SOURCE_PATHS:
        raise ValueError("Rolling checkpoints require the native Slime dataset cursor contract")


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _cursor_path(root, iteration):
    return root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"


def _delete_previous(root, iteration):
    """Delete just the previous managed temporary point, never scan a run."""
    try:
        checkpoint = root / f"iter_{iteration:07d}"
        record = _read_json(checkpoint / _COMPLETE)
        if not record or record.get("owner") != _OWNER or record.get("retain", True):
            return
        if checkpoint.is_symlink() or int((root / _TRACKER).read_text()) <= iteration:
            return
        shutil.rmtree(checkpoint)
        cursor = _cursor_path(root, iteration)
        cursor.unlink(missing_ok=True)
        logger.info("Removed previous rolling checkpoint %s", checkpoint)
    except Exception:
        # A failed cleanup must leave a usable new checkpoint, not stop RL.
        logger.exception("Could not remove previous rolling checkpoint %s", iteration)


def _model_finished(context):
    """The native rank-zero finalizer is the only latest-checkpoint publisher."""
    root = Path(context["root"])
    iteration = context["iteration"]
    checkpoint = root / f"iter_{iteration:07d}"
    tracker = root / _TRACKER
    previous = int(tracker.read_text()) if tracker.exists() else None
    _atomic_write(checkpoint / _COMPLETE, json.dumps(context, sort_keys=True) + "\n")
    _atomic_write(tracker, str(iteration) + "\n")
    logger.info("Committed recovery checkpoint after %s updates: %s (retain=%s)",
                iteration + 1, checkpoint, context["retain"])
    if previous is not None and previous != iteration:
        threading.Thread(target=_delete_previous, args=(root, previous), daemon=False).start()


def _make_schedule(original):
    @wraps(original)
    def schedule(request):
        context = _active_save.get()
        if context is None:
            return original(request)
        import torch.distributed as dist

        is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        finalizers = []
        replaced = 0
        for fn in request.finalize_fns:
            if (getattr(fn, "__name__", None) == "iter_finalize_fn"
                    and getattr(fn, "__module__", None) == "megatron.training.checkpointing"):
                finalizers.append(lambda context=context: _model_finished(context))
                replaced += 1
            else:
                finalizers.append(fn)
        if replaced != int(is_rank_zero):
            raise RuntimeError("Unsupported Megatron async iteration-finalizer contract")
        return original(request._replace(finalize_fns=finalizers))
    return schedule


def _init_persistent_worker(init_fn, rank: int) -> None:
    """Call MCore 0.20's ``init_persistent_async_worker(rank)`` safely."""
    params = inspect.signature(init_fn).parameters
    if any(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for p in params.values()
    ):
        init_fn(rank)
    else:
        init_fn()


def _make_save_model(original):
    @wraps(original)
    def save_model(self, rollout_id, force_sync=False):
        if not enabled(self.args):
            return original(self, rollout_id, force_sync=force_sync)
        validate_args(self.args)
        if not getattr(self, "_rolling_async_initialized", False):
            from megatron.training.async_utils import init_persistent_async_worker, is_empty_async_queue
            if not is_empty_async_queue():
                raise RuntimeError("Cannot initialize rolling writer with outstanding checkpoint saves")
            # MCore 0.20 requires the rank argument on init_persistent_async_worker(rank).
            rank = int(getattr(self.args, "rank", 0))
            _init_persistent_worker(init_persistent_async_worker, rank)
            self._rolling_async_initialized = True
        from megatron.training.async_utils import maybe_finalize_async_save

        # Native save_model already waits for a lagging predecessor. Close its
        # writer here as well, before allocating the next checkpoint snapshot.
        maybe_finalize_async_save(blocking=True, terminate=True)
        # Upstream Slime 79aa41a8 always uses the global rollout dataset cursor.
        if getattr(self.args, "rank", 0) == 0 and getattr(self, "rollout_manager", None) is not None:
            import ray
            ray.get(self.rollout_manager.save.remote(rollout_id))
        interval = self.args.rolling_checkpoint_retain_interval
        context = {
            "owner": _OWNER, "root": str(Path(self.args.save).absolute()),
            "iteration": rollout_id, "completed_updates": rollout_id + 1,
            "retain": force_sync or (rollout_id + 1) % interval == 0,
            "retain_interval": interval, "requires_cursor": True,
            "optimizer": True, "rng": True,
        }
        token = _active_save.set(context)
        try:
            # Native save snapshots tensors before returning and allows one
            # background writer. A lagging predecessor applies backpressure;
            # final saving drains before the actors are disposed.
            result = original(self, rollout_id, force_sync=force_sync)
            if force_sync:
                maybe_finalize_async_save(blocking=True, terminate=True)
            return result
        finally:
            _active_save.reset(token)
    return save_model


def _finalize_completed_save():
    from megatron.training.async_utils import is_empty_async_queue, maybe_finalize_async_save

    pending = not is_empty_async_queue()
    maybe_finalize_async_save(blocking=False)
    if is_empty_async_queue():
        # The idle native worker retains its last request's tensor buffers.
        # Close only after completion; native scheduling recreates it on demand.
        maybe_finalize_async_save(blocking=False, terminate=True)
        if pending:
            logger.info("Completed checkpoint writer closed; snapshot buffers released")


def _make_train(original):
    @wraps(original)
    def train(self, *args, **kwargs):
        if enabled(self.args):
            # All ranks poll at the same boundary. Background threads never
            # run NCCL or mutate model parameters.
            _finalize_completed_save()
        result = original(self, *args, **kwargs)
        if enabled(self.args):
            # Commit a writer that finished during this update before the
            # next rollout weight synchronization can fail.
            _finalize_completed_save()
        return result
    return train


def _make_data_save(original):
    @wraps(original)
    def save(self, rollout_id):
        # Slime's driver also calls save after save_model. Keep the cursor
        # captured before model saving; this actor handles calls serially.
        rolling = enabled(self.args)
        if rolling and rollout_id <= getattr(self, "_rolling_saved_rollout_id", -1):
            return
        result = original(self, rollout_id)
        if rolling:
            self._rolling_saved_rollout_id = rollout_id
        return result
    return save


class _SnapshotPlanner:
    """Freeze CPU tensors directly into their final IPC storage, one item at a time."""

    def __init__(self, planner):
        self.planner = planner
        self.cpu_bytes = 0

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def resolve_data(self, item):
        import torch

        tensor = self.planner.resolve_data(item)
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
            # Allocate final, tightly sized shared storage before copying so
            # neither native view handling nor multiprocessing copies it again.
            frozen = torch.empty_like(tensor, memory_format=torch.contiguous_format).share_memory_()
            frozen.copy_(tensor.detach())
            self.cpu_bytes += frozen.numel() * frozen.element_size()
            return frozen
        return tensor


def _make_prepare_write_data(original):
    @wraps(original)
    def prepare_write_data(self, plan, planner):
        if _active_save.get() is None:
            return original(self, plan, planner)
        snapshot = _SnapshotPlanner(planner)
        result = original(self, plan, snapshot)
        logger.info("Rolling checkpoint frozen CPU tensors: %.2f GiB in shared storage",
                    snapshot.cpu_bytes / (1024 ** 3))
        return result
    return prepare_write_data


def _resolve_async_filesystem_writer():
    """Return MCore 0.20's nvrx ``FileSystemWriterAsync`` (falling back to mcore if absent)."""
    try:
        from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
            FileSystemWriterAsync,
        )
        return FileSystemWriterAsync
    except ImportError:
        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync,
        )
        return FileSystemWriterAsync


def patch():
    global _applied
    if _applied:
        return
    from megatron.training import checkpointing
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    from slime.data.data_source import RolloutDataSource

    writer_cls = _resolve_async_filesystem_writer()
    checkpointing.schedule_async_save = _make_schedule(checkpointing.schedule_async_save)
    MegatronTrainRayActor.save_model = _make_save_model(MegatronTrainRayActor.save_model)
    MegatronTrainRayActor.train = _make_train(MegatronTrainRayActor.train)
    RolloutDataSource.save = _make_data_save(RolloutDataSource.save)
    writer_cls.prepare_write_data = _make_prepare_write_data(writer_cls.prepare_write_data)
    _applied = True
