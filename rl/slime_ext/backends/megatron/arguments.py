"""Patch for ``slime.backends.megatron_utils.arguments``.

Upstream validates/derives Megatron args from a flat HF config. Multimodal
models like Qwen3-Omni nest the language config under
``thinker_config.text_config``. This teaches the validation and default logic
to unwrap that, and to tolerate a missing ``shared_expert_intermediate_size``.

Replacements are injected into the upstream module so lazy call sites
(``from slime.backends.megatron_utils.arguments import megatron_parse_args``)
pick them up. Idempotent.
"""

_applied = False


def _with_slime_omni_arguments(extra_args_provider):
    """Compose Omni-only Megatron arguments with Slime's normal provider."""

    def provider(parser):
        if extra_args_provider is not None:
            updated_parser = extra_args_provider(parser)
            if updated_parser is not None:
                parser = updated_parser

        performance_group = None

        def group():
            nonlocal performance_group
            if performance_group is None:
                performance_group = parser.add_argument_group("slime_omni performance")
            return performance_group

        if not any("--response-only-log-probs" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--response-only-log-probs",
                action="store_true",
                default=False,
                help="Compute policy log-probability, entropy, and top-p replay only on response rows.",
            )
        if not any("--response-only-lm-head" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--response-only-lm-head",
                action="store_true",
                default=False,
                help=(
                    "Project only packed response rows through the Qwen3-Omni LM head; "
                    "requires response-only log-probs; supports packed two-chunk context parallelism."
                ),
            )
        if not any("--use-turn-level-gspo" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--use-turn-level-gspo",
                action="store_true",
                default=False,
                help=(
                    "Enable Turn-Level GSPO (TG-GSPO): compute per-turn geometric-mean "
                    "importance ratios using turn_ids and optional turn_weights."
                ),
            )
        if not any("--multimodal-attn-impl" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--multimodal-attn-impl",
                choices=("auto", "eager", "sdpa", "flash_attention_2"),
                default="auto",
                help=(
                    "Select the Hugging Face attention implementation for the frozen "
                    "Qwen3-Omni vision and audio towers."
                ),
            )
        if not any("--vision-batch-invariant-positions" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--vision-batch-invariant-positions", action="store_true", default=False,
                help="Use Qwen's fixed four-corner interpolation sum in the Omni vision tower.",
            )
        if not any("--vision-cp" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--vision-cp", action="store_true", default=False,
                help="Split frozen Omni vision encoding over CP ranks and gather all DeepStack features.",
            )
        if not any("--audio-fixed-conv-batch" in action.option_strings for action in parser._actions):
            group().add_argument(
                "--audio-fixed-conv-batch", action="store_true", default=False,
                help="Pad Omni audio chunks to conv_chunksize batches before HF's masked encoding.",
            )
        if not any("--rolling-checkpoint-retain-interval" in action.option_strings for action in parser._actions):
            checkpoint_group = parser.add_argument_group("slime_omni checkpoints")
            checkpoint_group.add_argument(
                "--rolling-checkpoint-retain-interval", type=int, default=0,
                help=("Keep every N completed updates permanently and only the latest other "
                      "checkpoint; zero disables rolling retention. Requires --async-save."),
            )
        return parser

    return provider


def validate_omni_args(args):
    """Validate Omni-specific argument invariants."""
    if getattr(args, "response_only_lm_head", False):
        if not getattr(args, "response_only_log_probs", False):
            raise ValueError("--response-only-lm-head requires --response-only-log-probs")
        if getattr(args, "allgather_cp", False):
            raise ValueError("--response-only-lm-head supports packed two-chunk CP, not allgather CP")
        if getattr(args, "enable_mtp_training", False):
            raise ValueError("--response-only-lm-head is not compatible with MTP training")
        if getattr(args, "use_score_centering", False):
            raise ValueError("--response-only-lm-head is not compatible with --use-score-centering")
    if getattr(args, "use_turn_level_gspo", False):
        estimator = getattr(args, "advantage_estimator", None)
        if estimator is not None and estimator != "gspo":
            raise ValueError(
                f"--use-turn-level-gspo requires --advantage-estimator gspo, got {estimator!r}"
            )


def _get_language_hf_config(hf_config):
    """Return the language sub-config used for Megatron argument validation."""
    if hf_config is None:
        return None

    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        return text_config

    thinker_config = getattr(hf_config, "thinker_config", None)
    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is not None:
        return thinker_text_config

    return hf_config


def patch():
    global _applied
    if _applied:
        return

    from copy import copy

    import slime.backends.megatron_utils.arguments as mod

    original_validate = mod._hf_validate_args
    original_defaults = mod._set_default_megatron_args
    original_parse = mod.megatron_parse_args
    original_validate_args = mod.validate_args

    def _hf_validate_args(args, hf_config):
        language = _get_language_hf_config(hf_config)
        # HF uses zero and MCore can use None for an absent shared expert.
        # Normalize a copy so neither the checkpoint config nor other users change.
        shared = getattr(language, "shared_expert_intermediate_size", None)
        expected = getattr(args, "moe_shared_expert_intermediate_size", None)
        if hasattr(language, "shared_expert_intermediate_size") and shared in (None, 0) and expected in (None, 0):
            language = copy(language)
            language.shared_expert_intermediate_size = expected
        original_validate(args, language)
        if getattr(args, "vocab_size", None) is None:
            args.vocab_size = getattr(language, "vocab_size", None)

    def _set_default_megatron_args(args, hf_config=None):
        if getattr(args, "vocab_size", None) is None and hf_config is not None:
            args.vocab_size = getattr(_get_language_hf_config(hf_config), "vocab_size", None)
        return original_defaults(args)

    def megatron_parse_args(extra_args_provider, skip_hf_validate=False):
        args = original_parse(
            _with_slime_omni_arguments(extra_args_provider),
            skip_hf_validate=skip_hf_validate,
        )
        validate_omni_args(args)
        return args

    def validate_args(args, *positional, **kwargs):
        result = original_validate_args(args, *positional, **kwargs)
        validate_omni_args(args)
        from slime_ext.backends.megatron.rolling_checkpoint import validate_args as validate_checkpoints
        validate_checkpoints(args)
        return result

    mod._get_language_hf_config = _get_language_hf_config
    mod._hf_validate_args = _hf_validate_args
    mod._set_default_megatron_args = _set_default_megatron_args
    mod.set_default_megatron_args = _set_default_megatron_args
    mod.validate_args = validate_args
    mod.megatron_parse_args = megatron_parse_args
    _applied = True
