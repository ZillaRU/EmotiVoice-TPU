"""
This code is modified from https://github.com/espnet/espnet.
"""

import torch
import torch.nn as nn
import numpy as np
from .. import EngineOV
from models.prompt_tts_modified.modules.encoder import Encoder
from models.prompt_tts_modified.modules.variance import DurationPredictor, VariancePredictor
from models.prompt_tts_modified.modules.alignment import AlignmentModule, GaussianUpsampling, viterbi_decode, average_by_duration
from models.prompt_tts_modified.modules.initialize import initialize

class PromptTTS(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.encoder = EngineOV('./model_file/tts/jit_am_encoder_1-512-384_1-1-512.bmodel')

        self.decoder = EngineOV('./model_file/tts/onnx_am_decoder_1-2048-384.bmodel')

        self.duration_predictor = EngineOV('./model_file/tts/am_durationpred-1_512_384-1_512_1.bmodel')

        self.pitch_predictor = EngineOV('./model_file/tts/am_pitchpred-1_512_384-1_512_1.bmodel')

        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config.model.encoder_n_hidden,
                kernel_size=config.model.variance_embed_kernel_size, #pitch_embed_kernel_size=1 in paddlespeech fs2
                padding=(config.model.variance_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(config.model.variance_embde_p_dropout), #pitch_embed_dropout=0.0
        )
        self.energy_predictor = EngineOV('./model_file/tts/am_energypred-1_512_384-1_512_1.bmodel')

        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config.model.encoder_n_hidden,
                kernel_size=config.model.variance_embed_kernel_size,
                padding=(config.model.variance_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(config.model.variance_embde_p_dropout),
        )

        self.length_regulator = GaussianUpsampling()
        self.alignment_module = AlignmentModule(config.model.encoder_n_hidden, config.n_mels)

        self.to_mel = nn.Linear(
            in_features=config.model.decoder_n_hidden, 
            out_features=config.n_mels,
        )

        self.spk_tokenizer = nn.Embedding(config.n_speaker, config.model.encoder_n_hidden)  # torch.from_numpy(np.load('./model_file/spk_tokenizer.npy'))  # nn.Embedding(config.n_speaker, config.model.encoder_n_hidden)
        self.src_word_emb =  nn.Embedding(config.n_vocab, config.model.encoder_n_hidden)  # torch.from_numpy(np.load('./model_file/src_word_emb.npy')) # nn.Embedding(config.n_vocab, config.model.encoder_n_hidden)
        self.embed_projection1 = nn.Linear(config.model.encoder_n_hidden * 2 + config.model.bert_embedding * 2, config.model.encoder_n_hidden)
        
        model_para_dict = torch.load('./model_file/tts/am_rest_weight.pth')
        self.load_my_state_dict(model_para_dict)
    
    # inputs_ling [1, seq_len] max:512
    # inputs_style_embedding [1, 768]
    # inputs_content_embedding [1,768]
    def forward(self, inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding , inputs_content_embedding, mel_targets=None, output_lengths=None, pitch_targets=None, energy_targets=None, alpha=1.0):
        B = inputs_ling.size(0)
        T = inputs_ling.size(1)
        _T = 512
        inputs_ling_pad = torch.cat((inputs_ling, torch.zeros((1, 512-inputs_ling.shape[1]), dtype=torch.int64)), 1)
        token_embed_pad = self.src_word_emb(inputs_ling_pad)
        src_mask = self._get_mask_from_lengths(input_lengths)
        x = self.encoder(
            [token_embed_pad.numpy().astype(np.float32),
            (~src_mask.unsqueeze(-2)).numpy().astype(np.float32)]
        )[0] ############
        x = torch.from_numpy(x[~src_mask]).unsqueeze(0)
        
        speaker_embedding = self.spk_tokenizer(inputs_speaker)
        x = torch.concat([x, speaker_embedding.unsqueeze(1).expand(B, T, -1), inputs_style_embedding.unsqueeze(1).expand(B, T, -1), inputs_content_embedding.unsqueeze(1).expand(B, T, -1)], dim=-1)
        x = self.embed_projection1(x)
        x = x.numpy()
        x = np.concatenate([x, np.zeros((1, _T-x.shape[1], x.shape[-1]))], axis=1).astype(np.float32)
        temp_src_mask = src_mask.unsqueeze(-1).numpy().astype(np.float32)
        
        p_outs = torch.from_numpy(self.pitch_predictor([x, temp_src_mask])[0])
        e_outs = torch.from_numpy(self.energy_predictor([x, temp_src_mask])[0])
        d_outs = torch.from_numpy(self.duration_predictor([x, temp_src_mask])[0])
        d_outs = torch.clamp(torch.round(d_outs.exp() - 1.0), min=0).long()
        
        src_mask = src_mask[:,:input_lengths[0]]
        x = x[:, :input_lengths[0], :]
        p_outs = p_outs[:,:input_lengths[0]]
        e_outs = e_outs[:,:input_lengths[0]]
        d_outs = d_outs[:,:input_lengths[0]]

        p_embs = self.pitch_embed(p_outs.unsqueeze(1)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.unsqueeze(1)).transpose(1, 2)
        x = torch.from_numpy(x) + p_embs + e_embs
        x = self.length_regulator(x, d_outs, None, ~src_mask)
        
        x_pad = torch.cat((x, torch.zeros((x.shape[0], 2048-x.shape[1], x.shape[2]))), axis=1)
        x_pad = torch.from_numpy(self.decoder([x_pad.numpy().astype(np.float32)])[0])
        x = x_pad[:, :x.shape[1], :]
        x = self.to_mel(x)
        return x

    def _get_mask_from_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_len = 512
        ids = (
            torch.arange(0, max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

    def get_mask_from_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_len = torch.max(lengths).item()
        ids = (
            torch.arange(0, max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask
    
    def average_utterance_prosody(
        self, u_prosody_pred: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        lengths = ((~src_mask) * 1.0).sum(1)
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                print(name, ":", param)
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(f"{name} is not loaded")

    def make_pad_mask(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).int()

        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
            batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


    def make_non_pad_mask(self, length, max_len=None):
        return ~self.make_pad_mask(length, max_len)


