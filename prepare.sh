#!/bin/bash
pip3 install --upgrade pip

# 定义两个包的 URL
TPU_PERF_AARCH64="https://github.com/radxa-edge/TPU-Edge-AI/releases/download/v0.1.0/tpu_perf-1.2.31-py3-none-manylinux2014_aarch64.whl"
TPU_PERF_X86_64="https://github.com/radxa-edge/TPU-Edge-AI/releases/download/v0.1.0/tpu_perf-1.2.31-py3-none-manylinux2014_x86_64.whl"

# 检查平台
case "$(uname -m)" in
    aarch64)
        WHL_URL=$TPU_PERF_AARCH64
        ;;
    x86_64)
        WHL_URL=$TPU_PERF_X86_64
        ;;
    *)
        echo "Unsupported platform"
        exit 1
        ;;
esac

# 安装 tpu-perf
if ! python3 -c "import tpu_perf" &> /dev/null; then
    echo "tpu_perf could not be found, installing..."
    pip3 install "$WHL_URL"
fi
pip3 install -r requirements.txt

python3 -m nltk.downloader averaged_perceptron_tagger_eng
sudo apt-get install libsndfile1 -y
mkdir -p ~/.cache/torch/hub/
mv assets/master.zip ~/.cache/torch/hub/
cd ~/.cache/torch/hub/
unzip master.zip
rm master.zip
mv snakers4-silero-vad-6c8d844 snakers4_silero-vad_master
echo "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" >> ~/.bashrc
source ~/.bashrc
cd -
