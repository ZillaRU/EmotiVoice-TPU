import torch
import numpy as np
import re
import soundfile
import commons
import os
import librosa
from mel_processing import spectrogram_torch
from converter import SynthesizerTrn
from models import EngineOV
import utils


class OpenVoiceBaseClass(object):
    def __init__(self, 
                config_path, 
                device='cuda:0'):
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = utils.get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, 'symbols', [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path, max_len=1024, quantize='f16'):
        checkpoint_dict = torch.load(os.path.join(ckpt_path,'checkpoint.pth'), map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        if os.path.exists(os.path.join(ckpt_path, f'decoder_{max_len}_{quantize}.bmodel')):
            self.model.dec = None
            del self.model.dec
            self.model.dec = EngineOV(os.path.join(ckpt_path, f'decoder_{max_len}_{quantize}.bmodel'), device_id=0)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print('missing/unexpected keys:', a, b)


class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if False:#kwargs.get('enable_watermark', True):
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None
        self.load_ckpt("./model_file/converter", max_len=2048)


    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]
        
        device = self.device
        hps = self.hps
        gs = []
        
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(device)
            y = y.unsqueeze(0)
            y = spectrogram_torch(y, hps.data.filter_length,
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                        center=False).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)

        return gs

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        hps = self.hps
        # load audio
        audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio).float()
        
        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
                        0, 0].data.cpu().float().numpy()
            # audio = self.add_watermark(audio, message)
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, hps.data.sampling_rate)
    
