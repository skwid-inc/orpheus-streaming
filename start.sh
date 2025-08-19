#!/bin/bash
set -e

export HF_HOME="/workspace/hf"
export TRANSFORMERS_OFFLINE=0

# Completely disable MPI
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_btl="^openib"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_MCA_plm_rsh_agent="sh -c"
export OMPI_MCA_orte_tmpdir_base="/tmp"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker0"

# Force single process mode for TensorRT-LLM
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Disable MPI entirely by setting this environment variable
export TRTLLM_DISABLE_MPI=1
export TENSORRT_LLM_USE_MPI=0

# Use single worker mode
export TRTLLM_SINGLE_WORKER=1

uvicorn main:app --host 0.0.0.0 --port 9090
