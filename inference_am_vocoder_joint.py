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
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import os, sys, warnings, torch, glob, argparse
import numpy as np
from models.hifigan.get_vocoder import MAX_WAV_VALUE
import soundfile as sf
from yacs import config as CONFIG
from tqdm import tqdm

def get_style_embedding(prompt_text, tokenizer, style_encoder):
    prompt = tokenizer([prompt_text], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]

    with torch.no_grad():
        # import pdb; pdb.set_trace()
        import time; st_time = time.time()
        output = style_encoder(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
        )
        print('====================== BERT time cost:', time.time()-st_time)
        style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding

def main(args, config):
    root_path = os.path.join(config.output_directory, args.logdir)

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)

    generator = JETSGenerator(conf)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}
    
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    
    text_path = args.test_file

    if os.path.exists(root_path + "/test_audio/audio/"):
        r = glob.glob(root_path + "/test_audio/audio/*")
        for j in r:
            os.remove(j)
    texts = []
    prompts = []
    speakers = []
    contents = []
    with open(text_path, "r") as f:
        for line in f:
            line = line.strip().split("|")
            speakers.append(line[0])
            prompts.append(line[1])
            texts.append(line[2].split())
            contents.append(line[3])
            
    for i, (speaker, prompt, text, content) in enumerate(tqdm(zip(speakers, prompts, texts, contents))):

        style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
        # import pdb; pdb.set_trace()
        content_embedding = get_style_embedding(content, tokenizer, style_encoder)

        if speaker not in speaker2id:
            continue
        speaker = speaker2id[speaker]

        text_int = [token2id[ph] for ph in text]
        
        sequence = torch.from_numpy(np.array(text_int)).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)]))
        style_embedding = torch.from_numpy(style_embedding).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).unsqueeze(0)
        speaker = torch.from_numpy(np.array([speaker]))
        with torch.no_grad():
            import time; st_time = time.time()
            infer_output = generator(
                    inputs_ling=sequence, # [1, xxx]
                    inputs_style_embedding=style_embedding, # [1, 768]
                    input_lengths=sequence_len, # [xxx]
                    inputs_content_embedding=content_embedding, # [1, 768]
                    inputs_speaker=speaker, # [x]
                    alpha=1.0
                )
            print('====================== Generator time cost:', time.time()-st_time)
            audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            sf.write(f"{i+1}.wav", data=audio, samplerate=config.sampling_rate) #h.sampling_rate


if __name__ == '__main__':
    print("run!")
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--logdir', type=str, required=True)
    p.add_argument("-c", "--config_folder", type=str, required=True)
    # p.add_argument("--checkpoint", type=str, required=False, default='', help='inference specific checkpoint, e.g --checkpoint checkpoint_230000')
    p.add_argument('-t', '--test_file', type=str, required=True, help='the absolute path of test file that is going to inference')

    args = p.parse_args() 
    ##################################################
    sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "/" + args.config_folder)

    from config import Config
    config = Config()
    ##################################################
    main(args, config)


