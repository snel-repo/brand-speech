import gc
import librosa
import logging
import nltk
nltk.download('punkt')
import numpy as np
np.random.seed(0)
import phonemizer
import pygame
import random
random.seed(0)
import re
import signal
import sys
import time
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import torchaudio
import os
import whisper
import yaml

from brand import BRANDNode

from nltk.tokenize import word_tokenize, sent_tokenize

from StyleTTS2.models import load_F0_models, load_ASR_models, build_model
from StyleTTS2.utils import recursive_munch
from StyleTTS2.text_utils import TextCleaner
from StyleTTS2.Utils.PLBERT.util import load_plbert
from StyleTTS2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# pygame requires a display
try:
    with open(os.path.join(os.path.expanduser('~'), '.DISPLAY'), 'r') as f:
        os.environ["DISPLAY"] = f.read().splitlines()[0]
except FileNotFoundError:
    if 'DISPLAY' not in os.environ:
        logging.error('No display found, exiting')
        sys.exit(1)

LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']

class tts_node_v2(BRANDNode):

    def __init__(self):

        super().__init__()

        self.input_stream = str(self.parameters.get('input_stream', 'tts_final_decoded_sentence'))
        self.model_path = str(self.parameters.get('model_path', '/samba/t15/T15_TTS_Model/version8-StyleTTS2'))
        self.ref_audio_path = str(self.parameters.get('ref_audio_path', '/samba/t15/T15_TTS_Model/version8-StyleTTS2/wavs/audio1.wav_0.wav'))
        self.volume = float(self.parameters.get('volume', 1.0))
        self.verbose = bool(self.parameters.get('verbose', False))
        self.alpha = float(self.parameters.get('alpha', 0.9))
        self.beta = float(self.parameters.get('beta', 0.9))
        self.diffusion_steps = int(self.parameters.get('diffusion_steps', 10))
        self.embedding_scale = float(self.parameters.get('embedding_scale', 2.0))
        self.speed = float(self.parameters.get('speed', 1.0))
        self.fs = int(self.parameters.get('fs', 24000))
        self.whisper_model_size = str(self.parameters.get('whisper_model_size', 'base'))
        self.device = str(self.parameters.get('device', 'cuda:0')) #'cpu', 'cuda', 'cuda:0'
        self.gpu_number = str(self.parameters.get('gpu_number', '0'))

        logging.info(f'Attempting to use device: {self.device}')

        self.r.set('tts_currently_playing', 0)
        
        # Init TTS 
        logging.info('Intitializing StyleTTS...')
        self.tts = stts2(
                model_path=self.model_path,
                ref_audio_path=self.ref_audio_path,
                whisper_model_size = self.whisper_model_size,
                device = self.device,
                fs = self.fs,
            )
        
        # initializing pygame
        logging.info('Initializing pygame...')
        pygame.mixer.init(frequency=self.fs, channels=1)

        logging.info('TTS initialized and ready to go.')

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)


    #play the audio(.wav)generated by the tts from memory  
    def play_audio_from_mem(self, wav, rate):
        start_play_time = time.time()
        # normalize wav to -1 to 1
        wav = wav / np.max(np.abs(wav))
        wav = (wav * self.volume * np.iinfo(np.int16).max).astype(np.int16)
        sound = pygame.sndarray.make_sound(wav)
        sound.play()

        while(time.time() - start_play_time < np.shape(wav)[0] / rate):
            time.sleep(0.001)


    def run(self):

        last_entry_seen = "0"
        timeout = 1000
        
        while True:
          
            redis_read = self.r.xread({self.input_stream: last_entry_seen}, block=timeout, count=1)

            if len(redis_read) != 0:

                entries = redis_read[0][1]

                for entry_id, entry_dict in entries:

                    last_entry_seen = entry_id
                    btt_output_text = entry_dict[b'final_decoded_sentence']

                    text_for_tts = btt_output_text.decode()
                                       
                    if text_for_tts.strip():
                        text_for_tts = text_for_tts.strip()

                        # check for spelling mode. add commas between letters if in spelling mode.
                        # spelling_mode = True
                        # for item in self.tts.clean_text(text_for_tts).split():
                        #     if item not in LETTERS:
                        #         spelling_mode = False
                        # if spelling_mode:
                            # text_for_tts = text_for_tts.replace(' ',', ')

                        # do TTS inference
                        self.r.set('tts_currently_playing', 1)
                        start = time.time()
                        wav = self.tts.inference(
                                    text             = text_for_tts,
                                    ref_s            = self.tts.ref_s,
                                    alpha            = self.alpha,
                                    beta             = self.beta,
                                    diffusion_steps  = self.diffusion_steps,
                                    embedding_scale  = self.embedding_scale,
                                    speed            = self.speed,
                                )
                        rtf = (time.time() - start) / (len(wav) / self.fs)

                        if self.verbose:
                            logging.info(f"TTS inference time = {rtf:5f}")
                            logging.info(f'Playing TTS audio: {text_for_tts}')

                        # play TTS audio
                        self.play_audio_from_mem(wav, self.fs)

                self.r.set('tts_currently_playing', 0)
                self.r.xadd('tts_node_info', {'tts_play_complete': str(True)})
                                               
            #no text was received via redis stream from brain_to_text node
            else:
                if self.verbose:
                    logging.info("No text has been received in last {0} ms".format(timeout))    


    def terminate(self, sig, frame):
        logging.info('SIGINT received, Exiting')
        gc.collect()
        sys.exit(0)

LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']

# make the phonemizer stop whining about mismatches
logging.getLogger('phonemizer').setLevel(logging.CRITICAL)

class stts2():
    def __init__(self, model_path, ref_audio_path, fs=24000, whisper_model_size='base', device='cuda:0'):

        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.whisper_model_size = whisper_model_size
        self.device = device
        self.fs = fs
        self.config = yaml.safe_load(open(os.path.join(self.model_path, "config.yml")))

        self.textcleaner = TextCleaner()

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.to_mel_mean, self.to_mel_std = -4, 4

        if self.device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

        # load whisper model
        self.whisper_model = whisper.load_model(self.whisper_model_size, device=self.device)

        # load phonemizer
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True, words_mismatch='ignore')

        # load pretrained ASR model
        ASR_config = self.config.get('ASR_config', False)
        ASR_path = self.config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = self.config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

        # load model params
        self.model_params = recursive_munch(self.config['model_params'])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]


        # find model checkpoint
        files = [f for f in os.listdir(self.model_path) if f.endswith('.pth')]
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # load model checkpoint
        print(sorted_files[-1])
        params_whole = torch.load(os.path.join(self.model_path, sorted_files[-1]), map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

        # diffusion sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

        # load ref audio
        self.ref_s = self.compute_style(self.ref_audio_path)


    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.to_mel_mean) / self.to_mel_std
        return mel_tensor


    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=self.fs)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.fs:
            audio = librosa.resample(audio, sr, self.fs)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)
    

    def inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, speed=1.0):

        if len(text) > 400:
            print('Input text too long. Breaking into sentences...')
            sentences = sent_tokenize(text)

            # make sure no individual sentence is too long. if it is, split by commas into subsentences
            for s, sent in enumerate(sentences):
                if len(sent) > 400:
                    sub_sent = sent.split(', ')
                    sub_sent = [t + ',' for t in sub_sent[:-1]] + [sub_sent[-1]]

                    sentences[s:s+1] = sub_sent

            # group sentences together and keep each group under 512 characters
            new_sentences = ['']
            i = 0
            for sent in sentences:
                if len(new_sentences[i]) + len(sent) < 400:
                    if new_sentences[i] == '':
                        new_sentences[i] += sent
                    else:
                        new_sentences[i] += ' ' + sent
                else:
                    i += 1
                    new_sentences.append(sent)
            sentences = new_sentences

        else:
            sentences = [text]

        all_wav = []
        for sentence in sentences:
            try:
                wav = self._infer(sentence, ref_s, alpha, beta, diffusion_steps, embedding_scale, speed)
                all_wav.append(wav)
            except:
                print('There was an error generating TTS for this sentence.')
                pass
        wav = np.concatenate(all_wav, axis=-1)

        return wav
    

    def _infer(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, speed=1.0):

        if len(text) < 35:
            print('Input text too short. Padding with extra text.')
            text_for_inference = text + ' Here are extra filler words to make this longer.'
        else:
            text_for_inference = text

        text_for_inference = text_for_inference.strip()
        ps = self.global_phonemizer.phonemize([text_for_inference])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        wav = np.concatenate((out.squeeze().cpu().numpy()[..., :-100], np.zeros((6000,))), axis=-1)

        if text != text_for_inference:
            print('Trimming extra audio...')
            wav = self.trim_wav(text, text_for_inference, wav)

        return wav
    

    def trim_wav(self, target_text, target_text_padded, wav):

        spelling_mode = True
        for item in self.clean_text(target_text).split():
            if item not in LETTERS:
                spelling_mode = False

        # resample from 23500 hz to 16000 hz
        temp_wav = np.array(wav).astype(np.float32)
        temp_wav = torchaudio.transforms.Resample(self.fs, 16000)(torch.from_numpy(temp_wav))
        transcription = self.whisper_model.transcribe(temp_wav, word_timestamps=True)

        if spelling_mode and '-' in transcription['text']:
            transcription['text'] = transcription['text'].replace('-', ' ')

        all_word_timings = []
        for segment in transcription['segments']:
            for word in segment['words']:
                all_word_timings.append((word['word'], word['start'], word['end']))

        if self.calculate_wer(target_text_padded, transcription['text']) <= 0.4:
            start_time = 0

            ind1 = all_word_timings[len(self.clean_text(target_text).split())-1][2] - 0.15
            ind2 = all_word_timings[len(self.clean_text(target_text).split())][1] + 0.15
            snippett = running_mean(np.square(wav[int(ind1*self.fs):int(ind2*self.fs)]), 50)

            try:
                end_time = np.argmin(snippett) / self.fs + ind1
            except:
                end_time = -3.1
                print('ERROR: messed up with trimming somehow. Removing the last 3 seconds instead')

            wav = wav[int(start_time*self.fs):int(end_time*self.fs)]

        else:
            print('WER was too high! Using the whole audio clip minus 3 seconds.')
            print(target_text_padded)
            print(transcription['text'])
            print(self.calculate_wer(target_text_padded, transcription['text']))
            print(transcription['segments'][0]['words'])
            wav = wav[:int(-3.1*self.fs)]

        return wav
    

    def clean_text(self, text):
        # Remove punctuation
        text = re.sub(r'[^a-zA-Z\- \']', '', text)
        text = text.replace('--', '').lower()
        text = text.replace(" '", "'").lower()

        text = text.strip()
        text = ' '.join(text.split())

        return text


    def calculate_error_rates(self, r, h):
        """
        Calculation of WER or PER with Levenshtein distance.
        Works only for iterables up to 254 elements (uint8).
        O(nm) time ans space complexity.
        ----------
        Parameters:
        r : list of true words or phonemes
        h : list of predicted words or phonemes
        ----------
        Returns:
        Word error rate (WER) or phoneme error rate (PER) [int]
        ----------
        Examples:
        >>> calculate_wer("who is there".split(), "is there".split())
        1
        >>> calculate_wer("who is there".split(), "".split())
        3
        >>> calculate_wer("".split(), "who is there".split())
        3
        """
        # initialization
        d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion    = d[i][j-1] + 1
                    deletion     = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]


    def calculate_wer(self, true_sent, pred_sent):
        """
        Calculation of WER with Levenshtein distance.
        """
        true_sent = self.clean_text(true_sent).split()
        pred_sent = self.clean_text(pred_sent).split()
        return self.calculate_error_rates(true_sent, pred_sent) / len(true_sent)
    
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
      

if __name__ == "__main__":
    gc.disable()

    node = tts_node_v2()
    node.run()

    gc.collect()