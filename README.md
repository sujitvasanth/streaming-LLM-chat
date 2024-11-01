### Streaming-LLM-chat
![samplechat](https://github.com/sujitvasanth/streaming-LLM-chat/blob/main/samplechat.gif)


This is a transformers library application that allows you to choose a local LLM and run streaming inference on GPU.

it uses:

- Python: 3.8.10
- transformers library: 4.36.2
- transformers_stream_generator library

the models are assumed to be in oogabooga textgeneration ui folder

the openchat model is available at https://huggingface.co/

TheBloke/openchat-3.5-0106-GPTQ

sujitvasanth/TheBloke-openchat-3.5-0106-GPTQ

I recently had difficulty with GPTQ installation see below

If you are using PyTorch 2.0, you will need to install AutoGPTQ from source. Likewise if you have problems with the pre-built wheels, you should try building from source:

pip3 uninstall -y auto-gptq
git clone https://github.com/PanQiWei/AutoGPTQ
cd AutoGPTQ
git checkout v0.5.1
pip3 install .
