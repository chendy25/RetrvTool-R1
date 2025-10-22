#!/bin/bash
set -x

########## 环境变量 ##########
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# Ray配置（保持不变）
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

########## NCCL配置 ##########
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# ⚠️ 建议删除 lo，或者改为实际网卡名 (如 eth0 / ib0)
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_MIN_NCHANNELS=1
export NCCL_LAUNCH_MODE=PARALLEL
export OMP_NUM_THREADS=1

########## GPU分配逻辑 ##########
# ✅ 关键修复：每个 Slurm rank 只看到一张 GPU
if [ -n "$SLURM_LOCALID" ]; then
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
  export CUDA_VISIBLE_DEVICES=0,1  # fallback for local test
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

########## 其他配置 ##########
export WANDB_API_KEY=4de5bca8c9ea79a88941a2a8d6698b93822ce5e6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=1

########## 数据集配置 ##########
dataset_name=pixel_reasoner/PixelReasoner_RL_Data
train_data=[$(pwd)/data/${dataset_name}/test/testtrain.parquet]
val_data=[$(pwd)/data/${dataset_name}/test/testval.parquet]
model_name=Qwen/Qwen2.5-VL-7B-Instruct

########## 训练配置 ##########
rl_alg=grpo
n_gpus_per_node=2
n_nodes=1
n=2
batch_size=1
ppo_mini_batch_size=1
max_prompt_length=2048
max_response_length=1024
max_action_length=1024
max_obs_length=2048
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)
temperature=1.0
top_p=1.0
enable_agent=True
strategy="fsdp2"
action_stop_tokens='</tool_call>'
max_turns=2
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=pixel_reasoner

########## GPU与并行策略 ##########
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=1        # ✅ 改成1，避免多rank抢同卡
gpu_memory_utilization=0.5
do_offload=True
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
max_num_batched_tokens=5000
run_name_postfix="debug-single-gpu-fixed-chunked-prefill"

########## 生成运行名称 ##########
if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export VLLM_USE_V1=1
rollout_mode='async'

########## 工具服务器 ##########
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "pixel_reasoner" --workers_per_tool 1 &
server_pid=$!

echo "Server (pid=$server_pid) started at $tool_server_url"

########## 启动训练 ##########
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=1 \
    data.dataloader_num_workers=1 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1

########## 清理 ##########
pkill -P -9 $server_pid
kill -9 $server_pid
