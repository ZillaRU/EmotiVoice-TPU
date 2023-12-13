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

import re, jieba
from frontend_cn import preload_g2p_cn, re_digits
from frontend_en import preload_preprocess_english
from g2p_en import G2p
import os


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preload_g2p_cn_en(text):
    parts = re_english_word.split(text)
    tts_text = ["<sos/eos>"]
    chartype = ''
    for part in parts:
        if part == ' ' or part == '': continue
        if re_digits.match(part) and (chartype == 'cn' or chartype == '') or contains_chinese(part):
            if chartype == 'en':
                tts_text.append('eng_cn_sp')
            phoneme = preload_g2p_cn(jieba.cut(part))
            chartype = 'cn'
        elif re_english_word.match(part):
            if chartype == 'cn':
                tts_text.append('cn_eng_sp')
            phoneme = preload_preprocess_english(g2p, lexicon, part).replace(".", "")
            chartype = 'en'
        else:
            continue
        tts_text.append( phoneme.replace("[ ]", "").replace("<sos/eos>", "") )
    tts_text.append("<sos/eos>")
    return " ".join(tts_text)

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None


re_english_word = re.compile('([a-z\-\.\']+|\d+[\d\.]*)', re.I)
jieba.initialize()
ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()
