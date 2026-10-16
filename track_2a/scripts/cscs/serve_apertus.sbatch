#!/bin/bash
# Serve Apertus 1.5 8B with vLLM on one Clariden GH200 node (OpenAI-compatible API on port 8000).
#   sbatch serve_apertus.sbatch      then:  squeue --me   (note the node name, e.g. nid001234)
#SBATCH --job-name=apertus-vllm
#SBATCH --account=g238
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=apertus-vllm-%j.log

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before sbatch: export HF_TOKEN=hf_...}"
srun --environment=apertus-vllm \
  vllm serve swiss-ai/Apertus-v1.5-8B \
    --host 0.0.0.0 --port 8000 \
    --chat-template-content-format string \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.22
