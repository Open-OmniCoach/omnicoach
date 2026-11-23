#!/usr/bin/env bash
# Shared Qwen3-Omni-30B-A3B training launcher sourced by examples/*/run.sh.
# Invokes `python -m slime_ext.train --ray-address ... --ray-env-vars-json ...`
# without inline Python driver strings.

EXTRA_RAY_ENV_KEYS=()
DATA_PREPARE_ARGS=()

qwen3_omni_configure_agent_defaults() {
    ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
    ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-16}
    PACK_TOKENS=${PACK_TOKENS:-65536}
    LOG_PROBS_PACK_TOKENS=${LOG_PROBS_PACK_TOKENS:-24576}
    FLUSH_CACHE_INTERVAL=${FLUSH_CACHE_INTERVAL:-2}

    export SGLANG_VLM_CACHE_SIZE_MB=${SGLANG_VLM_CACHE_SIZE_MB:-4096}
    export SGLANG_MM_FEATURE_CACHE_MB=${SGLANG_MM_FEATURE_CACHE_MB:-4096}
    EXTRA_RAY_ENV_KEYS=(
        SGLANG_VLM_CACHE_SIZE_MB
        SGLANG_MM_FEATURE_CACHE_MB
    )
}

qwen3_omni_init_env() {
    SLIME_DIR=${SLIME_DIR:-${ROOT_DIR}/third_party/slime}
    MEGATRON_DIR=${MEGATRON_DIR:-${ROOT_DIR}/.deps/Megatron-LM}
    PYTHON_BIN=${PYTHON_BIN:-python3}
    DRY_RUN=${DRY_RUN:-0}
    TRAIN_MODULE=${TRAIN_MODULE:-slime_ext.train}

    TRIAL_ID=${TRIAL_ID:-$(date -u +%Y%m%dT%H%M%S%NZ)}
    OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/outputs/${EXAMPLE_NAME}/${TRIAL_ID}}
    if [ "${DRY_RUN}" = 1 ]; then
        INPUT_DATA_PATH=${DATA_PATH:-${DEFAULT_DATA_HINT}}
        MODEL_DIR=${INIT_MODEL_DIR:-${MODEL_DIR:-/path/to/Qwen3-Omni-30B-A3B}}
    else
        INPUT_DATA_PATH=${DATA_PATH:?${DATA_PATH_HELP}}
        MODEL_DIR=${INIT_MODEL_DIR:-${MODEL_DIR:?Set INIT_MODEL_DIR to HF checkpoint}}
    fi
    DATA_PATH_RESOLVED=${OUTPUT_DIR}/train.jsonl
    LOAD_DIR=${RESUME_DIR:-${MODEL_DIR}}
    RESOLVED_CONFIG=${OUTPUT_DIR}/resolved-config.yaml

    export PYTHONPATH="${ROOT_DIR}:${SLIME_DIR}:${MEGATRON_DIR}:${PYTHONPATH:-}"
    export RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1
    export SLIME_OMNI_LAZY_MULTIMODAL_H2D=1
    export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=slime_ext.backends.sglang
    export TOKENIZERS_PARALLELISM=false

    CP_SIZE=${CP_SIZE:-2}
    PACK_TOKENS=${PACK_TOKENS:-32768}
    LOG_PROBS_PACK_TOKENS=${LOG_PROBS_PACK_TOKENS:-16384}
    ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
    ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
    ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
    FLUSH_CACHE_INTERVAL=${FLUSH_CACHE_INTERVAL:-1}

    CONFIG_YAML=$("${PYTHON_BIN}" "${EXAMPLE_DIR}/configure.py" --base "${CONFIG_BASE}")

    local ray_keys=(
        PYTHONPATH
        RAY_ENABLE_ZERO_COPY_TORCH_TENSORS
        SLIME_OMNI_LAZY_MULTIMODAL_H2D
        SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE
        TOKENIZERS_PARALLELISM
    )
    if [ "${#EXTRA_RAY_ENV_KEYS[@]}" -gt 0 ] 2>/dev/null; then
        ray_keys+=("${EXTRA_RAY_ENV_KEYS[@]}")
    fi
    ENV_JSON=$("${PYTHON_BIN}" -c 'import json, os, sys; print(json.dumps({k: os.environ[k] for k in sys.argv[1:] if k in os.environ}))' "${ray_keys[@]}")
}

qwen3_omni_append_agent_args() {
    RECIPE_ARGS+=(
        --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
        --fully-async-group-prefetch-factor "${GROUP_PREFETCH_FACTOR:-2}"
        --custom-generate-function-path "${PACKAGE}.rollout.generate"
        --custom-rm-path "${PACKAGE}.reward.score"
        --custom-convert-samples-to-train-data-path "${PACKAGE}.reward.convert_samples_to_train_data"
        --custom-rollout-log-function-path "${PACKAGE}.reward.log_metrics"
        --multimodal-keys '{}'
        --rollout-max-context-len 65536 --rollout-max-prompt-len 61440 --rollout-max-response-len 4096
        --partial-rollout
        --advantage-estimator gspo
        --sglang-mem-fraction-static 0.65
        --sglang-cuda-graph-bs 1 2 3 4
        --sglang-max-running-requests 4
        --sglang-context-length 65536
        --sglang-chunked-prefill-size 32768
        --sglang-enable-metrics
        --sglang-enable-request-time-stats-logging
    )
    if [ "${DYNAMIC_SAMPLING:-1}" = 1 ]; then
        RECIPE_ARGS+=(--dynamic-sampling-filter-path "${PACKAGE}.reward.filter_group")
    else
        RECIPE_ARGS+=(--dynamic-sampling-filter-path "${PACKAGE}.reward.reject_runtime_failures")
    fi
}

qwen3_omni_run() {
    local train_args=(
        --actor-num-nodes "${ACTOR_NUM_NODES}"
        --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
        --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"

        --spec slime_ext.models.qwen3_omni.provider get_qwen3_omni_model_provider
        --disable-bias-linear --qk-layernorm --group-query-attention
        --num-attention-heads 32 --num-query-groups 4 --kv-channels 128
        --num-layers 48 --hidden-size 2048 --ffn-hidden-size 768
        --normalization RMSNorm --position-embedding-type mrope --mrope-section 24 20 20
        --norm-epsilon 1e-6 --rotary-percent 1.0 --rotary-base 1000000
        --swiglu --untie-embeddings-and-output-weights --vocab-size 152064
        --moe-ffn-hidden-size 768 --moe-router-score-function softmax
        --moe-token-dispatcher-type flex --moe-enable-deepep --moe-router-topk 8
        --moe-layer-freq 1 --num-experts 128 --moe-grouped-gemm
        --moe-token-drop-policy probs --moe-router-dtype fp64
        --moe-permute-fusion --moe-aux-loss-coeff 0

        --hf-checkpoint "${MODEL_DIR}" --load "${LOAD_DIR}" --ref-load "${MODEL_DIR}"
        --model-name qwen3_omni --update-weight-mode full --update-weight-transport nccl
        --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_BYTES:-268435456}"
        --flush-cache-interval "${FLUSH_CACHE_INTERVAL}"

        --custom-config-path "${RESOLVED_CONFIG}"
        --prompt-data "${DATA_PATH_RESOLVED}" --input-key prompt --label-key label
        --metadata-key metadata
        --rollout-shuffle --rollout-skip-special-tokens --rollout-seed "${ROLLOUT_SEED:-42}"
        --seed "${TRAIN_SEED:-1234}"
        --num-epoch "${NUM_EPOCH:-1}" --rollout-batch-size 32 --n-samples-per-prompt 8
        --rollout-temperature 0.6 --rollout-top-p 0.95 --rollout-top-k 20
        --global-batch-size 256 --balance-data --over-sampling-batch-size 40

        --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1
        --adam-beta1 0.9 --adam-beta2 0.98 --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
        --disable-grpo-std-normalization --use-rollout-logprobs
        --use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.001}" --kl-loss-type low_var_kl --kl-coef 0
        --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28

        --train-backend megatron --micro-batch-size 1
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
        --context-parallel-size "${CP_SIZE}" --cp-comm-type a2a
        --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
        --recompute-granularity full --recompute-method block --recompute-num-layers 48
        --log-probs-chunk-size 2048 --recompute-loss-function
        --use-dynamic-batch-size --balance-by-flops
        --max-tokens-per-gpu "$((PACK_TOKENS / CP_SIZE))"
        --log-probs-max-tokens-per-gpu "$((LOG_PROBS_PACK_TOKENS / CP_SIZE))"

        --use-distributed-post
        --rollout-num-gpus-per-engine 2 --sglang-server-concurrency 8
        --router-policy cache_aware --router-balance-abs-threshold 3
        --use-rollout-routing-replay --response-only-log-probs --response-only-lm-head
        --rollout-data-transport "${ROLLOUT_DATA_TRANSPORT:-object-store}"

        --custom-megatron-before-log-prob-hook-path slime_ext.backends.megatron.before_log_prob
        --custom-megatron-before-train-step-hook-path slime_ext.backends.megatron.before_train
        --vision-batch-invariant-positions --vision-cp --audio-fixed-conv-batch
        --multimodal-attn-impl flash_attention_2
        --train-env-vars "${ENV_JSON}"
        --attention-dropout 0 --hidden-dropout 0
        --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend fused
        --freeze-params-name-list 'thinker\.visual\.' 'thinker\.audio_model\.'
    )

    train_args+=("${RECIPE_ARGS[@]}")

    local save_interval=${SAVE_INTERVAL:-50}
    local latest_save_interval=${LATEST_SAVE_INTERVAL:-0}
    if [ "${save_interval}" != 0 ]; then
        if [ "${latest_save_interval}" != 0 ]; then
            train_args+=(
                --save "${OUTPUT_DIR}/checkpoint"
                --save-interval "${latest_save_interval}"
                --rolling-checkpoint-retain-interval "${save_interval}"
                --async-save
                --use-persistent-ckpt-worker
                --ckpt-format torch_dist
            )
        else
            train_args+=(
                --save "${OUTPUT_DIR}/checkpoint"
                --save-interval "${save_interval}"
                --ckpt-format torch_dist
            )
        fi
    fi
    train_args+=("$@")

    local ray_addr=${RAY_ADDRESS:-}
    if [ "${DRY_RUN}" = 1 ]; then
        ray_addr=${ray_addr:-auto}
    fi
    local command=(
        "${PYTHON_BIN}" -u -m "${TRAIN_MODULE}"
        --ray-address "${ray_addr}"
        --ray-env-vars-json "${ENV_JSON}"
        "${train_args[@]}"
    )

    if [ "${DRY_RUN}" = 1 ]; then
        printf '# Resolved config (%s):
%s
' "${RESOLVED_CONFIG}" "${CONFIG_YAML}" >&2
        printf '%q ' "${command[@]}"
        printf '
'
        exit 0
    fi

    mkdir -p "${OUTPUT_DIR}"
    printf '%s
' "${CONFIG_YAML}" > "${RESOLVED_CONFIG}"
    "${PYTHON_BIN}" -m "${PACKAGE}.data" --input "${INPUT_DATA_PATH}" --output "${DATA_PATH_RESOLVED}" ${DATA_PREPARE_ARGS[@]+"${DATA_PREPARE_ARGS[@]}"}
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to an existing Ray cluster}"
    cd "${SLIME_DIR}"
    "${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/driver.log"
}
