from flask import Flask, render_template, request, send_file, g, jsonify, send_from_directory
from frontend import g2p_cn_en, G2p, read_lexicon
import scipy.io.wavfile as wavf
import torch
from transformers import AutoTokenizer
import os, sys, io
import numpy as np
import copy
import base64
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from yacs import config as CONFIG
import soundfile as SF


os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)
lexicon = read_lexicon()
g2p = G2p()

MAX_WAV_VALUE = 32768.0

class EmotiVoicePipeline:
    def __init__(self, config):
        with open(config.model_config_path, 'r') as fin:
            conf = CONFIG.load_cfg(fin)
        
        conf.n_vocab = config.n_symbols
        conf.n_speaker = config.speaker_n_labels

        self.style_encoder = StyleEncoder(config)
        self.generator = JETSGenerator(conf)
        self.sampling_rate = config.sampling_rate
        with open(config.token_list_path, 'r') as f:
            self.token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}
        with open(config.speaker2id_path, encoding='utf-8') as f:
            self.speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)


    def __call__(self, content, emotion_prompt, speaker):
        def get_style_embedding(prompt_text, tokenizer, style_encoder):
            prompt = tokenizer([prompt_text], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            input_ids = prompt["input_ids"]
            token_type_ids = prompt["token_type_ids"]
            attention_mask = prompt["attention_mask"]

            with torch.no_grad():
                output = style_encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
                )
                style_embedding = output["pooled_output"].cpu().squeeze().numpy()
            return style_embedding
        
        style_embedding = get_style_embedding(emotion_prompt, self.tokenizer, self.style_encoder)
        content_embedding = get_style_embedding(content, self.tokenizer, self.style_encoder)

        if speaker not in self.speaker2id:
            return None
        speaker = self.speaker2id[speaker]

        phonemes = g2p_cn_en(content, g2p, lexicon).split()
        text_int = [self.token2id[ph] for ph in phonemes]
        
        sequence = torch.from_numpy(np.array(text_int)).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)]))
        style_embedding = torch.from_numpy(style_embedding).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).unsqueeze(0)
        speaker = torch.from_numpy(np.array([speaker]))

        with torch.no_grad():
            infer_output = self.generator(
                    inputs_ling=sequence,
                    inputs_style_embedding=style_embedding,
                    input_lengths=sequence_len,
                    inputs_content_embedding=content_embedding,
                    inputs_speaker=speaker,
                    alpha=1.0
                )
        audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
        audio = audio.numpy().astype('int16')
        return audio

@app.before_first_request
def load_model():
    from config import Config
    config = Config()
    pipeline = EmotiVoicePipeline(config)
    app.config['pipeline'] = pipeline
    print("register pipeline to app object.")


@app.route('/tts', methods=['POST'])
def process_data():
    '''
    请求示例：
    {
        "text_content": "测试一下吧！地球爆炸啦！", # 建议限制下字数，当前一次推理最多输出16秒音频
        "speaker": "3370", # 可支持的speaker列表见./data/youdao/text/speaker2，建议做成下拉选项
        "emotion":"惊讶"  # 可支持的任意文字，参考的emotion列表见./data/youdao/text/emotion，建议做成下拉选项
    }
    '''
    data = request.get_json()
    transcript = data.get('text_content')
    speaker = data.get('speaker')
    emotion = data.get('emotion')
    res_audio = app.config['pipeline'](transcript, emotion, speaker)
   
    buffer = io.BytesIO()
    wavf.write(buffer, app.config['pipeline'].sampling_rate, res_audio)

    res_audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    response = jsonify({'code':0,'audio_content': res_audio_b64})
    return response


if __name__ == "__main__":
    # engine setup
    app.run(debug=False, port=8717, host="0.0.0.0", threaded=False)
