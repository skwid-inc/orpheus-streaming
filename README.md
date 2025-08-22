# Orpheus TTS Server with Low Latency Streaming

- RTX-4090, cuda12.8
- 200ms ttfb on fp16 using vllm
- <160 ms ttfb on fp16 using trt-llm

## Docker Deployment

For Docker deployment on A100 GPUs, see the [deployment/](./deployment/) directory which contains all necessary files and documentation.



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


- download required NLTK data
  - `python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"`
  - why: required for text processing in the TTS pipeline

- quick verify
  - `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -c "from cuda import cuda; import tensorrt_llm; print('OK', tensorrt_llm.__version__)"`

- run server (TRT)
  - ensure `uvicorn` resolves to this env: `echo $PATH | grep "$CONDA_PREFIX/bin"`
  - start: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ./start.sh`
  - alt if PATH conflicts: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m uvicorn main:app --host 0.0.0.0 --port 9090`

- notes
  - first startup may spend time compiling kernels (PyTorch/TensorRT-LLM). Subsequent runs are faster.
  - if you previously installed different CUDA component wheels, re-run the cuda-python pin step.
