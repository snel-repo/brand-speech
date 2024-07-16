import redis
import argparse
import numpy as np
import lm_decoder
from datetime import datetime
import time
import sys
import os
import subprocess
import atexit
import logging
from enum import Enum
import tensorflow as tf

'''
Command string for normal chang lm:
/home/lm-pc/miniconda3/envs/tf-gpu-test-2/bin/python /home/lm-pc/brand/brand-modules/npl-davis/nodes/brainToText_closedLoop/language-model-standalone/language-model-standalone.py --lm_path /home/lm-pc/brand/LanguageModels/chang_lm_sil --acoustic_scale 0.8 --blank_penalty 2

Command string for normal open webtext lm:
/home/lm-pc/miniconda3/envs/tf-gpu-test-2/bin/python /home/lm-pc/brand/brand-modules/npl-davis/nodes/brainToText_closedLoop/language-model-standalone/language-model-standalone.py --lm_path /home/lm-pc/brand/LanguageModels/openwebtext_3gram_lm_sil --acoustic_scale 0.3 --blank_penalty 7 --nbest 500

Command string for giant language model:
/home/lm-pc/miniconda3/envs/tf-gpu-test-2/bin/python /home/lm-pc/brand/brand-modules/npl-davis/nodes/brainToText_closedLoop/language-model-standalone/language-model-standalone.py --lm_path /home/lm-pc/brand/LanguageModels/openwebtext_5gram_lm_sil --do_opt True --nbest 500 --rescore True --acoustic_scale 0.3 --blank_penalty 9 --alpha 0.5
'''

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',level=logging.DEBUG)

# class for tracking decoder state
class State(Enum):
    IDLE = 0
    DECODING = 1
    DONE = 2

def build_lm_decoder(model_path,
                    max_active=7000,
                    min_active=200,
                    beam=17.,
                    lattice_beam=8.,
                    acoustic_scale=1.5,
                    ctc_blank_skip_threshold=1.0,
                    length_penalty=0.0,
                    nbest=1):

    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest
    )

    TLG_path = os.path.join(model_path, 'TLG.fst')
    words_path = os.path.join(model_path, 'words.txt')
    G_path = os.path.join(model_path, 'G.fst')
    rescore_G_path = os.path.join(model_path, 'G_no_prune.fst')
    if not os.path.exists(rescore_G_path):
        rescore_G_path = ""
        G_path = ""
    if not os.path.exists(TLG_path):
        raise ValueError('TLG file not found at {}'.format(TLG_path))
    if not os.path.exists(words_path):
        raise ValueError('words file not found at {}'.format(words_path))

    decode_resource = lm_decoder.DecodeResource(
        TLG_path,
        G_path,
        rescore_G_path,
        words_path,
        ""
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

    return decoder


def build_opt(modelName='facebook/opt-6.7b', cacheDir=None, device='auto', load_in_8bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(modelName, cache_dir=cacheDir)
    model = AutoModelForCausalLM.from_pretrained(modelName, cache_dir=cacheDir,
                                                 device_map=device, load_in_8bit=load_in_8bit)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def rescore_with_gpt2(model, tokenizer, hypotheses, lengthPenalty):
    model_class = type(model).__name__
    if model_class.startswith('TF'):
        inputs = tokenizer(hypotheses, return_tensors='tf', padding=True)
        outputs = model(inputs)
        logProbs = tf.math.log(tf.nn.softmax(outputs['logits'], -1))
        logProbs = logProbs.numpy()
    else:
        import torch
        inputs = tokenizer(hypotheses, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logProbs = torch.nn.functional.log_softmax(outputs['logits'].float(), -1).numpy()

    newLMScores = []
    B, T, _ = logProbs.shape
    for i in range(B):
        n_tokens = np.sum(inputs['attention_mask'][i].numpy())

        newLMScore = 0.
        for j in range(1, n_tokens):
            newLMScore += logProbs[i, j - 1, inputs['input_ids'][i, j].numpy()]

        newLMScores.append(newLMScore - n_tokens * lengthPenalty)

    return newLMScores


def gpt2_lm_decode(model, tokenizer, nbest, acousticScale, lengthPenlaty, alpha,
                   returnConfidence=False):
    hypotheses = []
    acousticScores = []
    oldLMScores = []
    for out in nbest:
        hyp = out[0].strip()
        if len(hyp) == 0:
            continue
        hyp = hyp.replace('>', '')
        hyp = hyp.replace('  ', ' ')
        hyp = hyp.replace(' ,', ',')
        hyp = hyp.replace(' .', '.')
        hyp = hyp.replace(' ?', '?')
        hypotheses.append(hyp)
        acousticScores.append(out[1])
        oldLMScores.append(out[2])

    if len(hypotheses) == 0:
        logging.error('len(hypotheses) == 0')
        return ("", []) if not returnConfidence else ("", [], 0.)

    acousticScores = np.array(acousticScores)
    newLMScores = np.array(rescore_with_gpt2(model, tokenizer, hypotheses, lengthPenlaty))
    oldLMScores = np.array(oldLMScores)

    totalScores = (acousticScale * acousticScores) + ((1 - alpha) * oldLMScores) + (alpha * newLMScores)
    maxIdx = np.argmax(totalScores)
    bestHyp = hypotheses[maxIdx]

    nbest_out = []
    min_len = np.min((len(nbest), len(newLMScores), len(totalScores)))
    for i in range(min_len):
        nbest_out.append(';'.join(map(str,[nbest[i][0], nbest[i][1], nbest[i][2], newLMScores[i], totalScores[i]])))

    if not returnConfidence:
        return bestHyp, nbest_out
    else:
        totalScores = totalScores - np.max(totalScores)
        probs = np.exp(totalScores)
        return bestHyp, nbest_out, probs[maxIdx] / np.sum(probs)


def connect_to_redis_server(redis_ip, redis_port):
    try:
        # logging.info("Attempting to connect to redis...")
        redis_conn = redis.Redis(host=redis_ip, port=redis_port)
        redis_conn.ping()
    except redis.exceptions.ConnectionError:    
        logging.warning("Can't connect to redis server (ConnectionError).")
        return
    else:
        logging.info("Connected to redis.")
        return redis_conn
    

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)


# main function
def main(args):

    lm_path = args.lm_path
    gpuNumber = args.gpuNumber

    max_active = args.max_active
    min_active = args.min_active
    beam = args.beam
    lattice_beam = args.lattice_beam
    acoustic_scale = args.acoustic_scale
    ctc_blank_skip_threshold = args.ctc_blank_skip_threshold
    length_penalty = args.length_penalty
    nbest = args.nbest
    blank_penalty = args.blank_penalty

    do_opt = args.do_opt          # acoustic scale = 0.8, blank penalty = 7, alpha = 0.5
    opt_cache_dir = args.opt_cache_dir
    alpha = args.alpha
    rescore = args.rescore
    
    redis_ip = args.redis_ip
    redis_port = args.redis_port
    input_stream = args.input_stream
    partial_output_stream = args.partial_output_stream
    final_output_stream = args.final_output_stream

    # create a nice dict of params to put into redis
    lm_args = {
        'lm_path': lm_path,
        'acoustic_scale': acoustic_scale,
        'blank_penalty': blank_penalty,
        'nbest': nbest,
        'redis_ip': redis_ip,
        'redis_port': redis_port,
        'input_stream': input_stream,
        'partial_output_stream': partial_output_stream,
        'final_output_stream': final_output_stream,
        'do_opt': int(do_opt),
        'opt_cache_dir': opt_cache_dir,
        'alpha': alpha,
        'rescore': int(rescore)
    }

    REDIS_STATE = -1

    logging.info(f'Using GPU # {gpuNumber}')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuNumber
    # don't use 100% of GPU (so the computer display has a little to work with)
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    logging.info('Initializing language model decoder...')
    start_time = time.time()
    ngramDecoder = build_lm_decoder(lm_path,
                                    max_active=7000,
                                    min_active=200,
                                    beam=17.,
                                    lattice_beam=8.,
                                    acoustic_scale=acoustic_scale,
                                    ctc_blank_skip_threshold=1.0,
                                    length_penalty=0.0,
                                    nbest=nbest)
    logging.info(f'Language model successfully initialized in {(time.time()-start_time):0.4f} seconds.')

    if do_opt:
        logging.info(f"Building opt model...")
        start_time = time.time()
        lm, lm_tokenizer = build_opt(
            cacheDir=opt_cache_dir,
            load_in_8bit=True,
        )
        logging.info(f'OPT model successfully built in {(time.time()-start_time):0.4f} seconds.')

    logging.info(f'Attempting to connect to redis at ip={redis_ip} and port={redis_port}')
    # r = redis.Redis(host=redis_ip, port=redis_port)

    r = connect_to_redis_server(redis_ip, redis_port)
    while r is None:
        r = connect_to_redis_server(redis_ip, redis_port)
        if r is None:
            logging.warning(f'At startup, could not connect to redis server at {redis_ip}:{redis_port}. Trying again in 3 seconds...')
            time.sleep(3)

    r.set('lm_reset_flag', 1)
    r.set('lm_finalize_flag', 0)

    timeout_ms = 100
    oldStr = ''

    # main loop
    logging.info('Entering main loop...')
    while True:

        # try catch is to make sure we're connected to redis, and reconnect if not
        try:
            r.ping()

        except redis.exceptions.ConnectionError:
            if REDIS_STATE != 0:
                logging.error(f'Could not connect to the redis server at at {redis_ip}:{redis_port}! I will keep trying...')
            REDIS_STATE = 0
            time.sleep(1)
            continue

        else:
            if REDIS_STATE != 1:
                logging.info('Successfully connected to the redis server.')
                logits_last_entry_seen = get_current_redis_time_ms(r)
            REDIS_STATE = 1

            # check if we need to reset
            if r.get('lm_reset_flag') == b'1':
                # Reset the language model and then the reset flag, then move on to the next loop
                oldStr = ''
                ngramDecoder.Reset()

                p = r.pipeline()
                p.set('lm_reset_flag', 0)
                p.xadd('remote_lm_args', lm_args) # add args to redis server on every reset to make sure they're in there for each block.
                p.execute()

                logging.info('Reset the language model.')
                continue

            # check if we need to finalize
            if r.get('lm_finalize_flag') == b'1':
                # Finalize decoding, add the output to the output stream, and then move on to the next loop
                ngramDecoder.FinishDecoding()

                oldStr = ''

                # Rescore with unpruned LM
                if rescore:
                    startT = time.time()
                    ngramDecoder.Rescore()
                    logging.info('Rescore time: %.3f' % (time.time() - startT))

                decoded_final = ''
                if do_opt:
                    # Rescore with OPT
                    startT = time.time()
                    nbest_out = []
                    for d in ngramDecoder.result():
                        nbest_out.append([d.sentence, d.ac_score, d.lm_score])
                    
                    try:
                        decoded_final, nbest_redis = gpt2_lm_decode(lm,
                                                        lm_tokenizer,
                                                        nbest_out,
                                                        acoustic_scale,
                                                        alpha=alpha,
                                                        lengthPenlaty=0.0)
                        logging.info('OPT time: %.3f' % (time.time() - startT))

                    except Exception as e:
                        logging.error(f'During OPT rescore: {e}')
                        decoded_final = ngramDecoder.result()[0].sentence
                        nbest_redis = ''
                        

                elif len(ngramDecoder.result()) > 0:
                    # Otherwise just output the best sentence
                    decoded_final = ngramDecoder.result()[0].sentence
                    nbest_redis = ''
                
                logging.info(f'Final:  {decoded_final}')
                p = r.pipeline()
                if rescore and do_opt:
                    p.xadd(final_output_stream, {'lm_response_final': decoded_final, 'scoring': ';'.join(nbest_redis)})
                else:   
                    p.xadd(final_output_stream, {'lm_response_final': decoded_final})
                p.set('lm_finalize_flag', 0)
                p.execute()

                logging.info('Finalized the language model.')
                continue

            # ---------- The loop can only get down to here if reset and finalize flags are both 0 -----------

            # try to read logits from redis stream
            try:
                read_result = r.xread(
                    {input_stream: logits_last_entry_seen},
                    count = 1,
                    block = timeout_ms
                )
            except redis.exceptions.ConnectionError:
                if REDIS_STATE != 0:
                    logging.error(f'Could not connect to the redis server at at {redis_ip}:{redis_port}! I will keep trying...')
                REDIS_STATE = 0
                time.sleep(1)
                continue

            if (len(read_result) >= 1): 
                # --------------- Read input stream --------------------------------
                for entry_id, entry_data in read_result[0][1]:
                    logits_last_entry_seen = entry_id
                    logits = np.frombuffer(entry_data[b'logits'], dtype=np.float32)

                logits = np.expand_dims(logits, axis=0)
                logits = np.expand_dims(logits, axis=0)

                # --------------- Run language model -------------------------------
                lm_decoder.DecodeNumpy(ngramDecoder,
                                        logits[0],
                                        np.zeros_like(logits[0]),
                                        np.log(blank_penalty))

                # display partial decoded sentence if it exists
                if len(ngramDecoder.result()) > 0:
                    decoded_partial = ngramDecoder.result()[0].sentence
                    newStr = f'Partial: {decoded_partial}'
                    if oldStr != newStr:
                        logging.info(newStr)
                        oldStr = newStr
                else:
                    logging.info('Partial: [NONE]')
                    decoded_partial = ''
                # print(ngramDecoder.result())
                r.xadd(partial_output_stream, {'lm_response_partial': decoded_partial})

            else:
                # timeout if no data received for X ms
                # logging.warning(F'No logits came in for {timeout_ms} ms.')
                continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_path', type=str, help='Path of language model folder (e.g., the openwebtext_lm_sil folder)')
    parser.add_argument('--gpuNumber', type=str, default='0', help='GPU number to use')

    parser.add_argument('--max_active', type=int, default=7000, help='max_active param for LM')
    parser.add_argument('--min_active', type=int, default=200, help='min_active param for LM')
    parser.add_argument('--beam', type=float, default=17.0, help='beam param for LM')
    parser.add_argument('--lattice_beam', type=float, default=8.0, help='lattice_beam param for LM')
    parser.add_argument('--ctc_blank_skip_threshold', type=float, default=1., help='ctc_blank_skip_threshold param for LM')
    parser.add_argument('--length_penalty', type=float, default=0.0, help='length_penalty param for LM')
    parser.add_argument('--acoustic_scale', type=float, default=0.3, help='Acoustic scale for LM')
    parser.add_argument('--nbest', type=int, default=1, help='# of candidate sentences for LM decoding')
    parser.add_argument('--blank_penalty', type=float, default=7.0, help='Blank penalty for LM')

    parser.add_argument('--rescore', action='store_true', help='whether or not to rescore')
    parser.add_argument('--do_opt', action='store_true', help='Use the opt model?')
    parser.add_argument('--opt_cache_dir', type=str, default="/home/lm-pc/brand/huggingface", help='path to opt cache')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha value [0-1]: Higher = more weight on OPT rescore. Lower = more weight on LM rescore')

    parser.add_argument('--redis_ip', type=str, default='192.168.150.2', help='IP of the BRAND redis stream (string)')
    parser.add_argument('--redis_port', type=int, default=6379, help='Port of the BRAND redis stream (int)')
    parser.add_argument('--input_stream', type=str, default="lm_input", help='Input stream containing logits')
    parser.add_argument('--partial_output_stream', type=str, default="lm_output_partial", help='Output stream containing partial decoded sentences')
    parser.add_argument('--final_output_stream', type=str, default="lm_output_final", help='Output stream containing final decoded sentences')

    args = parser.parse_args()

    main(args)