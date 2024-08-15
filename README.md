# Run Emoti-OpenVoice on your Airbox
This repo is a SG2300x-adapted demo of EmotiVoice(@NetEase Youdao) and OpenVoice(@MyShell.ai).

## 0. Clone the repository.

`git clone https://github.com/ZillaRU/EmotiVoice-TPU.git`

## 1. Download bmodels.
- Download the bmodels for EmotiVoice.
```sh
cd EmotiVoice-TPU
python3 -m pip install dfn
python3 -m dfn --url https://disk.sophgo.vip/sharing/KymDuWLGw
unzip EmotiVoice.zip
mv EmotiVoice model_file/tts
rm EmotiVoice.zip
```
**if you have no access to sophgo disk, you can use [baidu disk](https://pan.baidu.com/s/192mBCj_FZbXhI_sCij0tSg?pwd=abox ) instead.**

- Download bmodels of OpenVoice tone color converter [here](https://drive.google.com/file/d/1ErVDiMFvTwRj649pyoJI7rRDAh5pTGVT/view?usp=drive_link) and run `tar zxfv checkpoints.tar.gz`, `mv checkpoints/converter model_file/converter`, `rm -rf checkpoints.tar.gz checkpoints`.

- The `model_file` directory should be organized as:
```
model_file
├── converter
│   ├── checkpoint.pth
│   ├── config.json
│   ├── decoder_1024_f16.bmodel
│   └── decoder_2048_f16.bmodel
├── simbert-base-chinese
│   ├── config.json
│   └── vocab.txt
└── tts
    ├── am_durationpred-1_512_384-1_512_1.bmodel
    ├── am_energypred-1_512_384-1_512_1.bmodel
    ├── am_pitchpred-1_512_384-1_512_1.bmodel
    ├── am_rest_weight.pth
    ├── bert_poolout1-768_1-512_1-512_1-512.bmodel
    ├── hifigan_1-80-1024_F16.bmodel
    ├── jit_am_encoder_1-512-384_1-1-512.bmodel
    └── onnx_am_decoder_1-2048-384.bmodel
```


## 2. Run web demo:
Install tpu-perf for SoC or PCIE mode.
- For SoC (Airbox)
```sh
pip3 install https://github.com/radxa-edge/TPU-Edge-AI/releases/download/v0.1.0/tpu_perf-1.2.31-py3-none-manylinux2014_aarch64.whl
```
- For PCIE
```sh
pip3 install https://github.com/radxa-edge/TPU-Edge-AI/releases/download/v0.1.0/tpu_perf-1.2.31-py3-none-manylinux2014_x86_64.whl
```
Then run the following script.
```sh
sudo apt-get install libsndfile1 -y
pip3 install torch==2.0.1 torchaudio==2.0.2
pip3 install numpy numba scipy transformers==4.26.1 librosa soundfile yacs g2p_en jieba pypinyin whisper_timestamped onnxruntime gradio==4.19.2
mv assets/master.zip ~/.cache/torch/hub/
cd ~/.cache/torch/hub/
unzip master.zip
rm master.zip
mv snakers4-silero-vad-6c8d844 snakers4_silero-vad_master
echo "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" >> ~/.bashrc
source ~/.bashrc
cd -
python3 demo_page.py
```
*The torch version higher than 2.3 might lead to an ERROR due to poor compatibility for ARM.*

## 3. Using OpenAi-like api
- Install the dependencies: `pip3 install fastapi pydub uvicorn[standard] pyrubberband`.
- Then, run `uvicorn openai_api:app --reload --host 0.0.0.0 --port [port_number]`, the service will be available at `hostip:port_number` in few seconds.
The way to call the service is exactly the same as using OpenAI's TTS service.

## For more detailed usage, please refer to the README from the original repo of [EmotiVoice](https://github.com/netease-youdao/EmotiVoice) and [OpenVoice](https://github.com/myshell-ai/OpenVoice).
