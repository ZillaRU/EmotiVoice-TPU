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

import re
import argparse
from string import punctuation
import numpy as np



def preload_preprocess_english(g2p, lexicon, text):
    phones = []
    words = list(filter(lambda x: x not in {"", " "}, re.split(r"([,;.\-\?\!\s+])", text)))

    for w in words:
        if w.lower() in lexicon:
            phones += [
                "[" + ph + "]" 
                for ph in lexicon[w.lower()]
            ]+["engsp1"]
        else:
            phone=g2p(w)
            if not phone:
                continue

            if phone[0].isalnum():
                phones += ["[" + ph + "]" if ph != ' ' else 'engsp1' for ph in phone]
            elif phone == " ":
                continue
            else:
                phones.pop() # pop engsp1
                phones.append("engsp4")
    if "engsp" in phones[-1]:
        phones.pop()

    mark = "." if text[-1] != "?" else "?"
    phones = ["<sos/eos>"] + phones + [mark, "<sos/eos>"]
    return " ".join(phones)
    
