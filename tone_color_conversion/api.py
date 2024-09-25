import torch
import numpy as np
import re
import soundfile
import os
import librosa
from models import EngineOV
from .mel_processing import spectrogram_torch
from .converter import SynthesizerTrn
from .utils import get_hparams_from_file


class OpenVoiceBaseClass(object):
    def __init__(self, 
                config_path, 
                device='cuda:0'):
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = get_hparams_from_file(config_path)

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


class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if False: #kwargs.get('enable_watermark', True):
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None
        # self.load_ckpt("./model_file/converter", max_len=2048)


    def extract_se(self, ref_wav_list, se_save_path=None, max_len=1024):
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
                _len = y.shape[2]
                if _len < max_len:
                    y = torch.cat([y, torch.zeros(1, y.shape[1], max_len-y.shape[2])], axis=2)
                else:
                    y = y[:,:,:max_len]
                    print(f'===== WARNING =====: Your input ({y.shape[2]}) exceeds the length limit ({max_len}). The output is incomplete.')
                y = y.transpose(1, 2)
                g = torch.from_numpy(self.model.ref_enc([y.numpy()])[0]).unsqueeze(-1)[:(max_len//4-_len//max_len)]
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
    
