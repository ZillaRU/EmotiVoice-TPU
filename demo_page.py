# Copyright 2023, YOUDAO
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
import os, glob
os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from yacs import config as CONFIG
import torch
import re
import soundfile as sf
from frontend import g2p_cn_en, G2p, read_lexicon
from config.config import Config
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from tone_color_conversion import ToneColorConverter, get_se
from transformers import AutoTokenizer
import base64
from pathlib import Path
import uuid

DEVICE = "cpu"
MAX_WAV_VALUE = 32768.0


config = Config()

def get_models():
    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    generator = JETSGenerator(conf)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}

    return (style_encoder, generator, tokenizer, token2id, speaker2id)

speakers = config.speakers
models = get_models()
g2p = G2p()
lexicon = read_lexicon()
tone_color_converter = ToneColorConverter(f'./model_file/converter/config.json', device=DEVICE)


def get_style_embedding(prompt_text, tokenizer, style_encoder):
    prompt = tokenizer([prompt_text], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]

    with torch.no_grad():
        import time; st_time = time.time()
        output = style_encoder(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
        )
        print('====================== BERT time cost:', time.time()-st_time)
        style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding


def tts(text_content, emotion, speaker, models, output_path):
    text = g2p_cn_en(text_content, g2p, lexicon)
    (style_encoder, generator, tokenizer, token2id, speaker2id) = models

    style_embedding = get_style_embedding(emotion, tokenizer, style_encoder)
    content_embedding = get_style_embedding(text_content, tokenizer, style_encoder)

    speaker = speaker2id[speaker]

    text_int = [token2id[ph] for ph in text.split()]
    
    sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = torch.from_numpy(content_embedding).to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

    with torch.no_grad():

        infer_output = generator(
                inputs_ling=sequence,
                inputs_style_embedding=style_embedding,
                input_lengths=sequence_len,
                inputs_content_embedding=content_embedding,
                inputs_speaker=speaker,
                alpha=1.0
            )

    audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    sf.write(file=output_path, data=audio, samplerate=config.sampling_rate)
    print(f"Save the generated audio to {output_path}")

def tts_only(text_content, speaker, emotion):
    res_wav = f'./temp/speaker{speaker}-{uuid.uuid4()}.wav'
    tts(text_content, emotion, speaker, models, res_wav)
    return res_wav

def predict(text_content, speaker, emotion, tgt_wav, agree):
    # initialize a empty info
    text_hint = ''
    # agree with the terms
    if agree == False:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return (
            text_hint,
            None,
            None,
        )

    if len(text_content) < 30:
        text_hint += f"[ERROR] Please give a longer text \n"
        gr.Warning("Please give a longer text")
        return (
            text_hint,
            None,
            None,
        )

    if len(text_content) > 200:
        text_hint += f"[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        gr.Warning(
            "Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo for your usage"
        )
        return (
            text_hint,
            None,
            None,
        )
    src_wav = f'./temp/src-{speaker}.wav'
    tts(text_content, emotion, speaker, models, src_wav)

    try:
        # extract the tone color features of the source speaker and target speaker
        source_se, audio_name_src = get_se(src_wav, tone_color_converter, target_dir='processed', vad=True)
        target_se, audio_name_tgt = get_se(tgt_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        text_hint += f"[ERROR] Get source/target tone color error {str(e)} \n"
        gr.Warning(
            "[ERROR] Get source/target tone color error {str(e)} \n"
        )
        return (
            text_hint,
            None,
            None,
        )

    save_path = './temp/output.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_wav, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)

    text_hint += f'''Get response successfully \n'''

    return (
        text_hint,
        src_wav,
        save_path
    )


def convert_only(src_wav, tgt_wav, agree):
    # initialize a empty info
    text_hint = ''
    # agree with the terms
    if agree == False:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return (
            text_hint,
            None
        )
    try:
        # extract the tone color features of the source speaker and target speaker
        source_se, audio_name_src = get_se(src_wav, tone_color_converter, target_dir='processed', vad=True)
        target_se, audio_name_tgt = get_se(tgt_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        text_hint += f"[ERROR] Get source/target tone color error {str(e)} \n"
        gr.Warning(
            "[ERROR] Get source/target tone color error {str(e)} \n"
        )
        return (
            text_hint,
            None,
            None,
        )

    src_path = src_wav

    save_path = f'./temp/output.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)

    text_hint += f'''Get response successfully \n'''

    return (
        text_hint,
        save_path
    )


description = """
# 😊😮😭Emoti Open Voice with AirBox💬
"""

with gr.Blocks(analytics_enabled=False) as demo:

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr.Markdown(description)
    with gr.Tab('TTS mode'):
        with gr.Row():
            with gr.Column():
                speaker_gr = gr.Dropdown(choices=speakers, label="Speaker ID (说话人)")
                emotion_gr = gr.Textbox(label="情感 (开心/悲伤)")
                content_gr = gr.Textbox(label="需要合成的文本")
                tts_button = gr.Button("生成", elem_id="send-btn", visible=True)
            with gr.Column():
                synthesize = gr.Audio(type="filepath")
                tts_button.click(tts_only, [content_gr, speaker_gr, emotion_gr], outputs=[synthesize])
    
    with gr.Tab('TTS + Conversion mode'):
        with gr.Row():
            with gr.Column():
                speaker_gr = gr.Dropdown(choices=speakers, label="原始说话人ID")
                content_gr = gr.Textbox(
                    label="文本内容",
                    info="One or two sentences at a time is better. Up to 200 text characters.",
                    value="You got a dream, you gotta protect it. People can't do something themselves, they wanna tell you you can't do it. If you want something, go get it. Period.",
                )
                emotion_gr = gr.Textbox(label="语调情感 (开心/悲伤/愤怒/惊讶/冷酷...)")
                ref_gr = gr.Audio(
                    label="目标音色",
                    #info="点击上传目标音色的音频文件",
                    type="filepath",
                )
                tos_gr = gr.Checkbox(
                    label="Agree",
                    value=True,
                    info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
                )

                tts_button = gr.Button("生成", elem_id="send-btn", visible=True)

            with gr.Column():
                out_text_gr = gr.Text(label="Info")
                src_audio_gr = gr.Audio(label="TTS Audio", autoplay=True)
                tgt_audio_gr = gr.Audio(label="Target Audio", autoplay=True)
                tts_button.click(predict, [content_gr, speaker_gr, emotion_gr, ref_gr, tos_gr], outputs=[out_text_gr, src_audio_gr, tgt_audio_gr])

    with gr.Tab('Conversion-only mode'):
        with gr.Row():
            with gr.Column():
                cvt_src_gr = gr.Audio(
                    label="Source Audio",
                    #info="Click on the ✎ button to upload your own source speaker audio",
                    type="filepath",
                )
                cvt_ref_gr = gr.Audio(
                    label="Reference Audio",
                    #info="Click on the ✎ button to upload your own target speaker audio",
                    type="filepath",
                )
                cvt_tos_gr = gr.Checkbox(
                    label="Agree",
                    value=True,
                    info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
                )
                cvt_button = gr.Button("转换", elem_id="send-btn", visible=True)
            with gr.Column():
                out_text_gr = gr.Text(label="Info")
                cvt_audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
                cvt_button.click(convert_only, [cvt_src_gr, cvt_ref_gr, cvt_tos_gr], outputs=[out_text_gr, cvt_audio_gr])


demo.queue()  
demo.launch(debug=True, show_api=True, share=False, server_name="0.0.0.0")
