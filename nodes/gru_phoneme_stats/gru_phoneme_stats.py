from brand import BRANDNode
import numpy as np
import time
import os
import logging
import signal
import gc
from pathlib import Path
from glob import glob
import sys
from g2p_en import G2p
import re

# from davis_brand.brain_to_text_utils.general_utils import sentence_to_phonemes, remove_punctuation, get_current_redis_time_ms, cer_and_per_and_wer

SIL_DEF = ['SIL']

def remove_punctuation(sentence):
        # Remove punctuation
        sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
        sentence = sentence.replace('--', '').lower()
        sentence = sentence.replace(" '", "'").lower()

        sentence = sentence.strip()
        sentence = ' '.join(sentence.split())

        return sentence

# Convert text to phonemes
def sentence_to_phonemes(thisTranscription, g2p_instance=None):
    if not g2p_instance:
        g2p_instance = G2p()

    # Remove punctuation
    thisTranscription = remove_punctuation(thisTranscription)
    
    phonemes = []
    if len(thisTranscription) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p_instance(thisTranscription):
            if p==' ':
                phonemes.append('SIL')

            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)

        #add one SIL symbol at the end so there's one at the end of each word
        phonemes.append('SIL')
    
    return phonemes, thisTranscription

def get_current_redis_time_ms(r):
        t = r.time()
        return int(t[0]*1000 + t[1]/1000)

def calculate_error_rates(r, h):
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

def cer_and_per_and_wer(decodedSentences, trueSentences, decodedPhonemes=None, truePhonemes=None, returnCI=False, returnIndividualStats=False):
    if decodedPhonemes is not None and truePhonemes is not None:
        CALCULATE_PER = True
    else:
        CALCULATE_PER = False

    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []

    if returnIndividualStats:
        wer_individual = []
        cer_individual = []

    if CALCULATE_PER:
        allPhonemeErr = []
        allPhoneme = []
        if returnIndividualStats:
            per_individual = []


    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = calculate_error_rates([c for c in trueSent], [c for c in decSent])

        trueWords = trueSent.split(" ")
        decWords = decSent.split(" ")
        nWordErr = calculate_error_rates(trueWords, decWords)

        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

        if returnIndividualStats:
            wer_individual.append(nWordErr / len(trueWords))
            cer_individual.append(nCharErr / len(trueSent))

        if CALCULATE_PER:
            decPhones = decodedPhonemes[x]
            truePhones = truePhonemes[x]

            nPhonemeErr = calculate_error_rates(truePhones, decPhones)

            allPhonemeErr.append(nPhonemeErr)
            allPhoneme.append(len(truePhones))

            if returnIndividualStats:
                per_individual.append(nPhonemeErr / len(truePhones))


    cer = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)
    if CALCULATE_PER:
        per = np.sum(allPhonemeErr) / np.sum(allPhoneme)

    if not returnCI:
        if CALCULATE_PER:
            if returnIndividualStats:
                return (cer, cer_individual), (per, per_individual), (wer, wer_individual)
            else:
                return cer, per, wer
        else:
            if returnIndividualStats:
                return (cer, cer_individual), (wer, wer_individual)
            else:
                return cer, wer
        
    else:
        allChar = np.array(allChar)
        allCharErr = np.array(allCharErr)
        allWord = np.array(allWord)
        allWordErr = np.array(allWordErr)

        nResamples = 10000
        resampledCER = np.zeros([nResamples,])
        resampledWER = np.zeros([nResamples,])

        if CALCULATE_PER:
            allPhoneme = np.array(allPhoneme)
            allPhonemeErr = np.array(allPhonemeErr)
            resampledPER = np.zeros([nResamples,])

        for n in range(nResamples):
            resampleIdx = np.random.randint(0, allChar.shape[0], [allChar.shape[0]])
            resampledCER[n] = np.sum(allCharErr[resampleIdx]) / np.sum(allChar[resampleIdx])
            resampledWER[n] = np.sum(allWordErr[resampleIdx]) / np.sum(allWord[resampleIdx])
            
            if CALCULATE_PER:
                resampledPER[n] = np.sum(allPhonemeErr[resampleIdx]) / np.sum(allPhoneme[resampleIdx])

        cerCI = np.percentile(resampledCER, [2.5, 97.5])
        werCI = np.percentile(resampledWER, [2.5, 97.5])

        if CALCULATE_PER:
            perCI = np.percentile(resampledPER, [2.5, 97.5])
            if returnIndividualStats:
                return (cer, cerCI[0], cerCI[1], cer_individual), (per, perCI[0], perCI[1], per_individual), (wer, werCI[0], werCI[1], wer_individual)
            else:
                return (cer, cerCI[0], cerCI[1]), (per, perCI[0], perCI[1]), (wer, werCI[0], werCI[1])
        else:
            if returnIndividualStats:
                return (cer, cerCI[0], cerCI[1], cer_individual), (wer, werCI[0], werCI[1], wer_individual)
            else:
                return (cer, cerCI[0], cerCI[1]), (wer, werCI[0], werCI[1])

class brainToText_stats(BRANDNode):
    def __init__(self):
        super().__init__()

        self.verbose = bool(self.parameters.get('verbose', False))

        self.trial_info_lastEntrySeen = 0
        self.final_decoded_sentence_lastEntrySeen = 0

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

    

    def run(self):

        all_true_sentences = []
        all_true_sentences_phonemes = []
        all_decoded_sentences = []
        all_decoded_sentences_phonemes = []
        trial_durations_ms = []
        n_decoded_words = []

        g2p_instance = G2p()
        
        while True:

            true_sentence = ''
            decoded_sentence = ''
            true_sentence_phonemes = []
            decoded_sentence_phonemes = []

            # Get trial info. Will not leave this while loop until we find a new trial
            while True:
                stream_entry = self.r.xread(
                    {'trial_info': self.trial_info_lastEntrySeen},
                    count=1,
                    block=1000
                    )

                if len(stream_entry) == 0:
                    continue

                for entry_id, entry_dict in stream_entry[0][1]:
                    self.trial_info_lastEntrySeen = entry_id

                    start = np.frombuffer(entry_dict[b'go_cue_redis_time'], dtype=np.uint64)[0]
                    end = np.frombuffer(entry_dict[b'trial_end_redis_time'], dtype=np.uint64)[0]
                    true_sentence = entry_dict[b'cue'].decode()

                    trial_paused = int(entry_dict[b'ended_with_pause'].decode())
                    trial_timed_out = int(entry_dict[b'ended_with_timeout'].decode())
                break

            # move on to next trial if latest trial was paused or timed out
            if trial_paused or trial_timed_out:
                self.trial_info_lastEntrySeen = get_current_redis_time_ms(self.r) + 2000
                self.final_decoded_sentence_lastEntrySeen = get_current_redis_time_ms(self.r) + 2000
                logging.info('Skipping stat calculation for trial because it was paused or timed out.')
                continue

            # Get final decoded sentence
            read_result = self.r.xread(
                    {'tts_final_decoded_sentence': self.final_decoded_sentence_lastEntrySeen},
                    count = 1,
                    block = 30000,
                    )
            
            # if no decoded sentence found within 30s, continue to next trial 
            if len(read_result) == 0:
                self.trial_info_lastEntrySeen = get_current_redis_time_ms(self.r) + 2000
                self.final_decoded_sentence_lastEntrySeen = get_current_redis_time_ms(self.r) + 2000
                logging.info('Skipping stat calculation for trial because no decoded sentence was found within 30s. ')
                continue

            for entry_id, entry_dict in read_result[0][1]:
                self.final_decoded_sentence_lastEntrySeen = entry_id
                decoded_sentence = entry_dict[b'final_decoded_sentence'].decode()


            # skip if decoded sentence is empty
            if (len(decoded_sentence.split()) == 0) or (decoded_sentence.strip() in ['a','i']):
                self.trial_info_lastEntrySeen = get_current_redis_time_ms(self.r) + 2000
                self.final_decoded_sentence_lastEntrySeen = get_current_redis_time_ms(self.r) + 2000
                logging.info('Skipping stat calculation for trial because decoded sentence is empty.')
                continue

            # remove punctuation from true and decoded sentences
            true_sentence = remove_punctuation(true_sentence)
            decoded_sentence = remove_punctuation(decoded_sentence)

            # convert true and decoded sentences to phonemes
            true_sentence_phonemes = sentence_to_phonemes(true_sentence, g2p_instance)[0]
            decoded_sentence_phonemes = sentence_to_phonemes(decoded_sentence, g2p_instance)[0]

            # append to lists
            all_true_sentences.append(true_sentence)
            all_true_sentences_phonemes.append(true_sentence_phonemes)
            all_decoded_sentences.append(decoded_sentence)
            all_decoded_sentences_phonemes.append(decoded_sentence_phonemes)
            trial_durations_ms.append(end - start)
            n_decoded_words.append(len(decoded_sentence.split(' ')))


            # calculate CER, PER, and WER
            cer, per, wer = cer_and_per_and_wer(all_true_sentences, all_decoded_sentences, all_true_sentences_phonemes, all_decoded_sentences_phonemes, returnCI=False)

            # log results
            if self.verbose:
                logging.info(f'True sentence:... "{true_sentence}"')
                logging.info(f'Decoded sentence: "{decoded_sentence}"')
            logging.info(f'Cumulative CER: {cer*100:0.2f}%, PER: {per*100:0.2f}%, WER: {wer*100:0.2f}%, WPM: {np.sum(n_decoded_words)/np.sum(trial_durations_ms)*1000*60:0.3f}, # of trials: {len(all_true_sentences)}')

            ## TODO: change WPM to time period from [first decoded word] to [button press]

            

    def terminate(self, sig, frame):
        logging.info('SIGINT received, Exiting')
        gc.collect()
        sys.exit(0)


if __name__ == "__main__":
    gc.disable()

    node = brainToText_stats()
    node.run()

    gc.collect()