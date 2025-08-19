# Orpheus TTS Server with Low Latency Streaming

- RTX-4090, cuda12.8
- 200ms ttfb on fp16 using vllm
- <160 ms ttfb on fp16 using trt-llm

## Docker Deployment

For Docker deployment on A100 GPUs, see the [deployment/](./deployment/) directory which contains all necessary files and documentation.

### installation
- `sudo apt-get -y install libopenmpi-dev`
- `conda create -n trt python=3.10 && conda activate trt` or use a virtual env with python3.10
- `pip install -r requirements.txt`
- update start.sh with your hf-token
- `bash start.sh`

### with runpod
- start a pod with RTX 4090 gpu and image `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- ensure port 9090 is open if you want to access externally
- `bash install.sh` or copy paste install.sh within the entrypoint of container

### Note: 
The hf repo of model `canopylabs/orpheus-3b-0.1-ft` has optimiser / fsdp files which are not required for inference. However trt-llm download all the files so ensure you have enough storage (~100GB) available in the pod / machine. Or you can stop the process midway (after tokeniser and model safetensors are downloaded) and cleanup these extra files and set `export TRANSFORMERS_OFFLINE=1` in start.sh and start the process again

### Clean install (TRT-LLM, tested fresh)

- create env
  - `conda create -n trt_clean python=3.10 -y`
  - `conda activate trt_clean`

- install pytorch (CUDA 12.4 wheels)
  - `python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0`

- install project/runtime deps
  - `python -m pip install fastapi "uvicorn[standard]" transformers==4.51.0 snac batched google-generativeai>=0.3.0 g2p_en`
  - `python -m pip install tensorrt==10.9.0.34 tensorrt_llm==0.19.0`

- pin cuda-python to avoid import issue
  - `python -m pip install --force-reinstall cuda-python==12.6.0`
  - why: newer 12.8/12.9 can break `from cuda import cuda` imports; see NVIDIA note [link](https://github.com/NVIDIA/cuda-python/issues/476)

- quick verify
  - `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -c "from cuda import cuda; import tensorrt_llm; print('OK', tensorrt_llm.__version__)"`

- run server (TRT)
  - ensure `uvicorn` resolves to this env: `echo $PATH | grep "$CONDA_PREFIX/bin"`
  - start: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ./start.sh`
  - alt if PATH conflicts: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m uvicorn main:app --host 0.0.0.0 --port 9090`

- notes
  - first startup may spend time compiling kernels (PyTorch/TensorRT-LLM). Subsequent runs are faster.
  - if you previously installed different CUDA component wheels, re-run the cuda-python pin step.
