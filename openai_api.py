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
import soundfile as sf
from fastapi import FastAPI, Response
from typing import Optional
from pydantic import BaseModel
from pydub import AudioSegment
from config import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


def load_model():
    pipeline = EmotiVoicePipeline(config)
    return pipeline


config = Config()
pipeline = load_model()
lexicon = read_lexicon()
g2p = G2p()
app = FastAPI()


class SpeechRequest(BaseModel):
    input: str
    voice: str = '8051'
    prompt: Optional[str] = ''
    language: Optional[str] = 'zh_us'
    model: Optional[str] = 'emoti-voice'
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1.0


@app.post("/v1/audio/speech")
def text_to_speech(speechRequest: SpeechRequest):
    np_audio = pipeline(speechRequest.input, speechRequest.prompt, speechRequest.voice)
    y_stretch = np_audio
    if speechRequest.speed != 1.0:
        y_stretch = pyrb.time_stretch(np_audio, config.sampling_rate, speechRequest.speed)
    wav_buffer = io.BytesIO()
    sf.write(file=wav_buffer, data=y_stretch,
             samplerate=config.sampling_rate, format='WAV')
    buffer = wav_buffer
    response_format = speechRequest.response_format
    if response_format != 'wav':
        wav_audio = AudioSegment.from_wav(wav_buffer)
        wav_audio.frame_rate=config.sampling_rate
        buffer = io.BytesIO()
        wav_audio.export(buffer, format=response_format)

    return Response(content=buffer.getvalue(),
                    media_type=f"audio/{response_format}")
