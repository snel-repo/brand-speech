from brand import BRANDNode
from enum import Enum
import numpy as np
import time
import tensorflow as tf
from datetime import datetime
import random
import os
import yaml
import logging
import coloredlogs
from omegaconf import OmegaConf
import signal
import gc
from pathlib import Path
from glob import glob
from tensorflow.python.training import py_checkpoint_reader
from scipy.interpolate import interp1d

import sys
import lm_decoder

from speech_utils.models.models import GRU
from speech_utils.utils.gaussSmooth import gaussSmooth

# class for tracking decoder state
class State(Enum):
    IDLE = 0
    DECODING = 1
    DONE = 2

task_state_to_legacy = {
    b'': -1,
    b'start_trial': 0,
    b'go': 1,
    b'freeze': 3,
    b'end_trial': 3
}

# class for normalization adaptation
class FeatureStats:
    def __init__(self, windowSize, mean, std, minSentences):
        self.windowSize = windowSize
        self.minSentences = minSentences
        self.buffer = []
        self.startMean = mean
        self.startStd = std
        self.mean = self.startMean
        self.std = self.startStd

    def update(self, features):
        # Remove the head if buffer is full
        if len(self.buffer) >= self.windowSize:
            self.buffer.pop(0)
        self.buffer.append(features)

        bufSize = len(self.buffer)
        if bufSize >= self.minSentences:
            self.mean = np.mean(np.concatenate(self.buffer, 0), 0)
            self.std = np.std(np.concatenate(self.buffer, 0), 0)
        else:
            new_mean = np.mean(np.concatenate(self.buffer, 0), 0)
            self.mean = (new_mean * bufSize + self.startMean * (self.minSentences - bufSize)) / self.minSentences
            new_std = np.std(np.concatenate(self.buffer, 0), 0)
            self.std = (new_std * bufSize + self.startStd * (self.minSentences - bufSize)) / self.minSentences

    def reset(self):
        self.buffer = []
        self.mean = self.startMean
        self.std = self.startStd


class brainToText_closedLoop(BRANDNode):
    def __init__(self):
        super().__init__()

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)


    def build_lm_decoder(self, model_path,
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
    

    def sort_models(self, model_list):
        # function to sort params list in ascending order of block number

        model_num = []
        for i in range(len(model_list)):
            model_num.append(int(model_list[i].split('rnn_model_')[1]))

        ind = np.argsort(np.array(model_num))
        model_list = list(np.array(model_list)[ind])

        return model_list


    def sort_blockmean_dirs(self, blockMean_dir_list):
        # function to sort blockmean dir list in ascending order of block number

        blockMean_dir_num = []
        for path in blockMean_dir_list:
            blockMean_dir_num.append(int(path.split('(')[1].split(')')[0]))

        ind = np.argsort(np.array(blockMean_dir_num))
        blockMean_dir_list = list(np.array(blockMean_dir_list)[ind])

        return blockMean_dir_list



    def rearrange_speech_logits(self, logits, has_sil=False):
        if not has_sil:
            logits = np.concatenate([logits[:, :, -1:], logits[:, :, :-1]], axis=-1)
        else:
            logits = np.concatenate([logits[:, :, -1:], logits[:, :, -2:-1], logits[:, :, :-2]], axis=-1)
        return logits
    

    def replace_words(self, sentence, word_substitutions):
        words = sentence.split(' ')
        for i, word in enumerate(words):
            if word in word_substitutions:
                words[i] = word_substitutions[word]

        sentence = ' '.join(words)
        return sentence
    

    def get_current_redis_time_ms(self):
        t = self.r.time()
        return int(t[0]*1000 + t[1]/1000)
    

    def reset_remote_lm(self):
        self.r.set('lm_reset_flag', 1)
        logging.info('Resetting remote language model before continuing...')

        # wait for remote language model to reset. Send a warning every 5 seconds if it takes too long.
        lm_reset_start_time = time.time()
        while self.r.get('lm_reset_flag') != b'0':
            time.sleep(0.01)
            if (time.time() - lm_reset_start_time) > 5:
                logging.error('Remote language model did not reset! Check lm-pc.')
                lm_reset_start_time = time.time()
            
        logging.info('Remote language model reset.')

        return
    

    def add_input_layer(self, inputLayers, normLayers, args):
        new_normLayer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[args['dataset']['nInputFeatures']])
        new_inputModel = tf.keras.models.clone_model(inputLayers[-1])
        new_inputModel.set_weights(inputLayers[-1].get_weights())

        logging.info(f'Adding norm and input layer {len(inputLayers)}, copying weights from {len(inputLayers)-1}')
        normLayers.append(new_normLayer)
        inputLayers.append(new_inputModel)

        return inputLayers, normLayers
        

    def readDecoderControlStream(self):
        stream = self.r.xrange('decoderControl:stream')
        if len(stream) == 0:
            return None

        if b'newModelPath' in stream[-1][1]:
            newModelPath = stream[-1][1][b'newModelPath'].decode('utf-8')
            return newModelPath
        return None


    # single decoding step function. @tf.function gives 4x speed improvement
    # puts data through normalization, smoothing, day-transform, and then RNN
    @tf.function
    def runSingleDecodingStep(self, normLayer, linearInputLayer, rnn, x, states, applySmooth=True, kernelSD=2):
        x = normLayer(x)
        if applySmooth:
            x = gaussSmooth(x, kernelSD=kernelSD, padding='VALID')
        x = linearInputLayer(x)
        logits, states = rnn(x, states, training=False, returnState=True)

        return logits, states

    def load_channel_mask(self):
        # initialize the channel mask to include all channels
        self.ch_mask = np.arange(self.n_features)
        # remove channels specified in excl_chans
        if self.excl_chans:
            self.ch_mask = np.setdiff1d(self.ch_mask, self.excl_chans)

        # get list of masked channels
        if hasattr(self, 'ch_mask_stream'):
            ch_mask_entry = self.r.xrevrange(self.ch_mask_stream,
                                             '+',
                                             '-',
                                             count=1)
            if ch_mask_entry:
                stream_mask = np.frombuffer(ch_mask_entry[0][1][b'channels'],
                                            dtype=np.uint16)
                self.ch_mask = np.intersect1d(self.ch_mask, stream_mask)
                for c in range(1, -(self.n_features // -self.tot_ch)):
                    self.ch_mask = np.concatenate((self.ch_mask, self.ch_mask + self.tot_ch * c))
                logging.info("Loaded channel mask from stream "
                             f"{self.ch_mask_stream}")
                if not self.zero_masked_chans:  # masked channels are dropped
                    logging.info('Overriding n_features parameter '
                                 f'{self.n_features} with {len(self.ch_mask)}')
                    self.n_features = len(self.ch_mask)
            else:
                logging.warning(
                    f"'ch_mask_stream' was set to {self.ch_mask_stream}, but "
                    "there were no entries. Defaulting to using all channels")
                self.ch_mask = np.arange(self.n_features)
        self.ch_mask.sort()
        logging.info(self.ch_mask)


    def run(self):

        p = self.r.pipeline()
        p.set('partial_decoded_sentence_current', '')
        p.set('final_decoded_sentence_current', '')
        p.set('b2t_decoder_initialized', 0)
        p.execute()

        ## Load parameters, using `self.parameters`.
        binned_input_stream = self.parameters["binned_input_stream"]
        output_stream = self.parameters["output_stream"]
        blockMean_path = self.parameters["blockMean_path"]
        blockMean_load_num = int(self.parameters["blockMean_load_num"])
        RNN_dir = self.parameters["RNN_path"]
        RNN_model_number = int(self.parameters.get("RNN_model_number", -1))
        LM_dir = self.parameters["LM_path"]
        logit_interpolation_factor = int(self.parameters.get('logit_interpolation_factor', 1))
        acousticScale = float(self.parameters["LM_acousticScale"])
        nBest = int(self.parameters["LM_nBest"])
        blankPenalty = int(self.parameters["LM_blankPenalty"])
        verbose = self.parameters["verbose"]
        input_network_num = int(self.parameters["input_network_num"])      
        gpuNumber = str(self.parameters.get("gpuNumber", "0"))       # GPU for tensorflow to use. -1 means that GPU is hidden and inference will happen on CPU.
        adaptMean = self.parameters.get("adaptMean", True)                # whether to adapt mean and std of normalization layer
        adaptWindowSize = int(self.parameters.get("adaptWindowSize", 20))      # how many (max) recent trials to use for normalization adaptation
        adaptMinSentences = int(self.parameters.get("adaptMinSentences", 10))  # how many (min) recent trials to use for normalization adaptation
        autosaveStats = self.parameters.get("autosaveStats", True)        # whether to save normalization stats to file
        autosaveStats_path = self.parameters.get("autosaveStats_path", blockMean_path)  # where to save normalization stats to file
        use_online_trainer = self.parameters.get("use_online_trainer", False)
        auto_punctuation = self.parameters.get("auto_punctuation", True)
        legacy = self.parameters.get("legacy_mode", True)
        task_state_stream = self.parameters.get("task_state_stream", 'task_state')
        sync_key = self.parameters.get("sync_key", 'sync').encode()
        time_key = self.parameters.get("time_key", 'ts').encode()

        use_local_lm = self.parameters.get("use_local_lm", True)

        self.n_features = int(self.parameters.get("n_features", 512))
        self.excl_chans = self.parameters.get("excl_chans", [])
        self.ch_mask_stream = self.parameters.get("ch_mask_stream", None)
        self.tot_ch = int(self.parameters.get("tot_ch", 256))
        self.zero_masked_chans = bool(self.parameters.get("zero_masked_chans", True))
        
        # prepare stuff for remote language model if needed
        if not use_local_lm:
            p = self.r.pipeline()
            p.set('lm_reset_flag', 0)
            p.set('lm_finalize_flag', 0)
            p.execute()

            remote_lm_input_stream = self.parameters.get("remote_lm_input_stream", 'lm_input')
            remote_lm_output_partial_stream = self.parameters.get("remote_lm_output_partial_stream", 'lm_output_partial')
            remote_lm_output_final_stream = self.parameters.get("remote_lm_output_final_stream", 'lm_output_final')

            remote_lm_output_partial_lastEntrySeen = self.get_current_redis_time_ms()
            remote_lm_output_final_lastEntrySeen = remote_lm_output_partial_lastEntrySeen

        # list of word substutitons (to minimize spelling errors)
        word_substitutions = {
            # 'u': 'you',
            # 'r': 'are',
            'ok': 'okay',
        }
        
        if auto_punctuation:
            self.punctuation_last_input_entry_seen = self.get_current_redis_time_ms()

        if logit_interpolation_factor>1:
            logging.info(f'Logits will be interpolated by a factor of: {logit_interpolation_factor}')

        # ------------------------------ find RNN model path ---------------------------------------
        if not os.path.exists(RNN_dir):
            logging.error(f'RNN directory not found: {RNN_dir}')

        if RNN_model_number != -1:
            # choose a specific model number
            RNN_model_name = sorted([str(x) for x in Path(RNN_dir).glob(f'rnn_model_{RNN_model_number}')])
            if len(RNN_model_name) == 0:
                logging.error(f'Could not find model number {RNN_model_number} in directory: {RNN_dir}')
            RNN_model_name = RNN_model_name[-1]
            
        else:
            # choose the latest model number
            RNN_model_name = self.sort_models([str(x) for x in Path(RNN_dir).glob('rnn_model_*')])
            if len(RNN_model_name) == 0:
                logging.error(f'Could not find any models in directory: {RNN_dir}')
            RNN_model_name = RNN_model_name[-1]


        ## load decoder args file
        args = OmegaConf.load(str(Path(RNN_model_name, 'args.yaml')))
        currentModelPath = RNN_model_name
        zScoreClip = 10
        bufferSize = args['smoothKernelSD'] * args['model']['patch_stride'] + args['model']['patch_size']


        # set GPU for tensorflow to use. -1 means that GPU is hidden and inference will happen on CPU.
        if gpuNumber=="-1":
            logging.info('I will run my RNN inference on the CPU.')
        else:
            logging.info(f'I will run my RNN inference on GPU #{gpuNumber}.')
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuNumber

        # don't use 100% of GPU (so the computer display has a little to work with)
        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)


        # ------------------------------ load blockMean and blockStd -------------------------------------
        
        if not os.path.exists(blockMean_path):
            # This is almost exclusively a block of code that needs to run for a
            # pretrained outside of session decoder running without collecting 
            # any data in session. 
            logging.warning(f"Folder at {blockMean_path} not found, trying to pull norms from thresh_norm means/stds")
            save_filepath = self.r.config_get('dir')['dir']
            save_filepath = os.path.dirname(save_filepath)  
            # This is supposedly where calcThreshNorm saves its info, so we can
            # by default pull norm and STD from here if the provided 
            # blockMean_path is not found.
            blockMean_path_tn = os.path.join(save_filepath, 'thresh_norm')
            if not os.path.exists(blockMean_path_tn):
                logging.error(f'Block Mean directory not found: {blockMean_path}\nAND thresh_norm directory not found: {blockMean_path_tn}\nPlease run the reference block.')

            save_filename = self.r.config_get('dbfilename')['dbfilename']
            save_filename = os.path.splitext(save_filename)[0]

            # Pathnames should be DIR/DIR/DIR/subjectid_date_blocknumber.yaml
            updated_mean_dirs = glob(str(Path(blockMean_path_tn, f'{save_filename}_*.yaml')))
            list.sort(updated_mean_dirs)
            blockMean_file = updated_mean_dirs[-1]
            data = yaml.safe_load(open(blockMean_file, 'r'))
            blockMean = np.array(data['means']) # len(channels)*2 because the second half of the list contains SBP
            blockStd = np.array(data['stds']) # len(channels)*2 because the second half of the list contains SBP
            logging.info(f'Loaded means and stds from: {blockMean_file}')
        else:
            updated_mean_dirs = glob(str(Path(blockMean_path, 'updated_means_block(*)')))

            if updated_mean_dirs == []:
                blockMean_path = blockMean_path
            elif blockMean_load_num == -1:
                blockMean_path = self.sort_blockmean_dirs(updated_mean_dirs)[-1]
            else:
                blockMean_path = glob(str(Path(blockMean_path, f'updated_means_block({blockMean_load_num})')))[-1]

            logging.info(F'Loading blockMean and blockStd from: {blockMean_path}')
            if os.path.isfile(f'{blockMean_path}/blockMean.npy') and os.path.isfile(f'{blockMean_path}/blockStd.npy'):
                blockMean = np.load(blockMean_path + "/blockMean.npy")
                blockStd = np.load(blockMean_path + "/blockStd.npy")
                logging.info(f'Loaded blockMean and blockStd from: {blockMean_path}')

            elif os.path.isfile(f'{blockMean_path}/rdbToMat_blockMean.npy') and os.path.isfile(f'{blockMean_path}/rdbToMat_blockStd.npy'):
                blockMean = np.load(blockMean_path + "/rdbToMat_blockMean.npy")
                blockStd = np.load(blockMean_path + "/rdbToMat_blockStd.npy")
                logging.info(f'Loaded rdbToMat_blockMean and rdbToMat_blockStd from: {blockMean_path}')
                
            else:
                logging.error(f'Could not find block means on path: {blockMean_path}')


        # --------------------------- load channel mask --------------------------------------------
        self.load_channel_mask()

        # --------------------------- initialize language model ------------------------------------
        if use_local_lm:
            # local lm
            logging.info(F'Loading language model from: {LM_dir}')
            ngramDecoder = self.build_lm_decoder(
                LM_dir,
                acoustic_scale = acousticScale,
                nbest = nBest
            )
        else:
            # remote lm
            logging.info(F'Using language model remotely from lm-pc')
            # self.reset_remote_lm()


        # ----------------------------- initialize GRU model -----------------------------------------
        logging.info(F'Initializing RNN.')
        gru_model = GRU(args['model']['nUnits'], 
                        args['model']['weightReg'], 
                        args['dataset']['nClasses']+1, 
                        args['model']['dropout'], 
                        args['model']['nLayers'], 
                        args['model']['patch_size'], 
                        args['model']['patch_stride'])
        gru_model(tf.keras.Input(shape=(None, args['model']['inputNetwork']['inputLayerSizes'][-1])))
        gru_model.trainable = False; #args['model']['trainable']
        print(gru_model.summary())

        # Build day transformation networks and normalization layers
        logging.info(F'Initializing input networks.')
        nInputLayers = len(args['dataset']['sessions'])
        inputLayers = []
        normLayers = []

        for layerIdx in range(nInputLayers):
            normLayer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[args['dataset']['nInputFeatures']])

            inputModel = tf.keras.Sequential()
            inputModel.add(tf.keras.Input(shape=(None, args['dataset']['nInputFeatures'])))
            inputModel.add(tf.keras.layers.Dense(args['model']['inputLayerSize'],
                                                    activation = args['model']['inputNetwork']['activation'],
                                                    kernel_initializer = tf.keras.initializers.identity(),
                                                    kernel_regularizer = tf.keras.regularizers.L2(args['model']['weightReg'])))
            inputModel.add(tf.keras.layers.Dropout(rate = args['model']['inputNetwork']['dropout']))
            inputModel.trainable = False; #args['model']['inputNetwork']['trainable']
            inputModel.summary()

            inputLayers.append(inputModel)
            normLayers.append(normLayer)


        # load pre-trained RNN
        logging.info(F'Loading pre-trained RNN from: {RNN_model_name}')
        ckptPath = tf.train.latest_checkpoint(RNN_model_name)
        ckptVars = {}
        ckptVars['net'] = gru_model
        for x in range(len(normLayers)):
            ckptVars['normLayer_'+str(x)] = normLayers[x]
            ckptVars['inputLayer_'+str(x)] = inputLayers[x]
        checkpoint = tf.train.Checkpoint(**ckptVars)
        checkpoint.restore(ckptPath).expect_partial()


        # add another layer and copy weights if we are using the online trainer
        if use_online_trainer: 
            logging.info(f'# input layers before adding one: {len(inputLayers)}')
            inputLayers, normLayers = self.add_input_layer(inputLayers, normLayers, args)
            logging.info(f'# input layers after adding one: {len(inputLayers)}')

        # check for new rnn model from online trainer
        if use_online_trainer:
            online_trainer_RNN_model_name = f"{RNN_model_name.split('rnn_model_')[0]}online_training_rnn_model_{RNN_model_name.split('rnn_model_')[1]}"
            ckptPath = tf.train.latest_checkpoint(online_trainer_RNN_model_name)

            if ckptPath is not None:
                logging.info(f'Loading new RNN checkpoint: {ckptPath}')
                loadLayerIdx = len(inputLayers) - 1
                ckptVars = {}
                ckptVars['net'] = gru_model
                ckptVars['normLayer_'+str(loadLayerIdx)] = normLayers[-1]
                ckptVars['inputLayer_'+str(loadLayerIdx)] = inputLayers[-1]
                checkpoint = tf.train.Checkpoint(**ckptVars)
                checkpoint.restore(ckptPath).expect_partial()
                # logging.info(f'Loaded new RNN checkpoint: {ckptPath}')

        # set decoder state to IDLE.
        decoderState = State.IDLE

        # counter variables
        blankClass = 0
        blankTicks = 0
        nonBlankTicks = 0
        frameCounter = 0
        taskState = 0

        # intialize stats for normalization adaptation
        stats = FeatureStats(adaptWindowSize, blockMean, blockStd, adaptMinSentences)
        self.r.xadd('brainToText_normStats', {'mean': np.array(stats.mean).tobytes(), 'std': np.array(stats.std).tobytes()})
        sentenceBuffer = []
        
        # initialaize recent data buffer
        recent_data = np.zeros([1, bufferSize, args['dataset']['nInputFeatures']], dtype=np.float32)

        # reset RNN states and warm up RNN
        logging.info(F'Warming up RNN.')
        states = [gru_model.initStates] + [None] * len(gru_model.rnnLayers)
        prev_logits = None
        self.runSingleDecodingStep(
            normLayers[input_network_num],
            inputLayers[input_network_num], 
            rnn=gru_model,
            x=tf.constant(recent_data, dtype=tf.float32),
            states=states,
            applySmooth=True,
            kernelSD=args['smoothKernelSD'])

        # reset LM
        if use_local_lm:
            ngramDecoder.Reset()
        else:
            self.reset_remote_lm()

        # figure out and create stats save path
        if autosaveStats:
            metadata_stream = self.r.xrange(b'block_metadata')
            block_num = int(metadata_stream[0][1].get(b'block_number', b'-1').decode())
            autosaveStats_path = str(Path(autosaveStats_path, f'updated_means_block({block_num})'))
            logging.info(f'I will save normalization stats to: {autosaveStats_path}')

        # prepare to read data from redis stream
        last_entry_seen = "$"
        timeout_ms = 2000

        # set redis var to tell sentence task that the decoder is initialized
        self.r.set('b2t_decoder_initialized', 1)
        logging.info(F'Starting decoding loop.')
        
        while True:
            # get one bin of data from redis stream
            read_result = self.r.xread(
                {binned_input_stream: last_entry_seen},
                count = 1,
                block = timeout_ms
            )

            # timeout if no data received for X ms
            if len(read_result) == 0:
                logging.warning(F'No binned data came in for {timeout_ms} ms.')
                continue

            # read this data snippet
            for entry_id, entry_dict in read_result[0][1]:
                last_entry_seen = entry_id

                # read and concatenate threshold crossings and spike band power
                data_snippet = np.frombuffer(bytearray(entry_dict[b'samples']), dtype=np.float32)
                # quick patch, convert summed SBP to mean SBP
                data_snippet[len(data_snippet)//2:] /= 20.

                sync_dict = entry_dict[sync_key]

            # get the task state [-1=INITIALIZING, 0=START, 1=GO, 3=END, 4=PAUSED]
            if legacy:
                newTaskState = int(self.r.get('task_state_current').decode())
            else:
                reply = self.r.xrevrange(task_state_stream, '+', '-', count=1)
                newTaskState = task_state_to_legacy[reply[0][1][b'state']]
            forceStop = False
            if newTaskState==-1:
                # Task has not been started yet
                taskState = newTaskState
                continue

            elif (taskState!=4 and newTaskState==4):
                # trial was just paused
                if decoderState == State.DECODING:
                    # decoder is still decoding and needs to be stopped
                    forceStop = True
                    if verbose:
                        logging.info(f'The trial was just paused and is still decoding. Forcing stop. Old task state = {taskState}, '\
                                    f'New task state = {newTaskState}, decoderState = {decoderState}, forceStop = {forceStop}')
                else:
                    # decoder was already stopped, we can continue
                    taskState = newTaskState
                    continue

            elif newTaskState==4:
                # the trial is still paused
                taskState = newTaskState
                continue

            elif (taskState!=0 and newTaskState==0):
                # We just entered the delay period.

                # check for new rnn model from online trainer
                if use_online_trainer:
                    newModelPath = self.readDecoderControlStream()
                    if newModelPath is not None:
                        if newModelPath != currentModelPath:
                            logging.info(f'Loading new RNN checkpoint: {newModelPath}')
                            loadLayerIdx = len(inputLayers) - 1
                            ckptVars = {}
                            ckptVars['net'] = gru_model
                            ckptVars['normLayer_'+str(loadLayerIdx)] = normLayers[-1]
                            ckptVars['inputLayer_'+str(loadLayerIdx)] = inputLayers[-1]
                            checkpoint = tf.train.Checkpoint(**ckptVars)
                            checkpoint.restore(newModelPath).expect_partial()
                            # logging.info(f'Loaded new RNN checkpoint: {newModelPath}')
                            currentModelPath = newModelPath

                if verbose:
                    logging.info(f'Delay period just started. Old task state = {taskState}, '\
                                f'New task state = {newTaskState}, decoderState = {decoderState}, forceStop = {forceStop}')
                taskState = newTaskState
                continue

            elif (taskState==0 and newTaskState==0):
                # We are still in the delay state. Go cue hasn't happened yet. Continue to wait until the go cue.
                taskState = newTaskState
                continue

            elif (taskState==0 and newTaskState==1):
                # Go cue just happened. Start decoding.
                decoderState = State.IDLE
                if verbose:
                    logging.info(f'GO period just started. Old task state = {taskState}, '\
                                f'New task state = {newTaskState}, decoderState = {decoderState}, forceStop = {forceStop}')
                    
            elif (taskState==1 and newTaskState==1):
                # We are still in the GO period.
                pass

            elif (taskState!=3 and newTaskState==3):
                # Trial just ended. Stop decoding.
                forceStop = True
                if verbose:
                    logging.info(f'Trial just ended. Forcing stop. Old task state = {taskState}, '\
                                f'New task state = {newTaskState}, decoderState = {decoderState}, forceStop = {forceStop}')

            taskState = newTaskState

            # If decoder is done decoding, there's nothing more to do except to wait for the next trial to start.
            if decoderState == State.DONE:
                continue

            # append data snippet to buffer for normalization adaptation
            sentenceBuffer.append(data_snippet)
            # logging.info('I just appended data to the sentence buffer!')

            # normalize data snippet
            data_snippet = (data_snippet - stats.mean) / (stats.std + 1e-8)   # z score
            data_snippet[data_snippet > zScoreClip] = 0                     # z score clip
            data_snippet[data_snippet < -zScoreClip] = 0
            # zero out masked channels
            if self.zero_masked_chans:
                data_snippet_zeroed = np.zeros_like(data_snippet)
                data_snippet_zeroed[self.ch_mask] = data_snippet[self.ch_mask]
                data_snippet = data_snippet_zeroed
            else:
                data_snippet = data_snippet[self.ch_mask]
            data_snippet = data_snippet[np.newaxis, np.newaxis, :]          # expand axis

            # concatenate data snippet to buffer
            recent_data = np.concatenate([recent_data[:,1:,:], data_snippet], axis=1)   # delete oldest data snippet, add newest

            # increment frame counter
            frameCounter += 1

            if forceStop or (frameCounter >= bufferSize):
                if (blankTicks == 0) and (decoderState == State.IDLE):
                    if verbose:
                        logging.info(F'Starting decoding!')
                    decoderState = State.DECODING
                    self.r.xadd(output_stream, {'start': last_entry_seen})

                if decoderState == State.DECODING:
                    if not forceStop and ((frameCounter - bufferSize) % args['model']['patch_stride'] != 0):
                        # if we don't meet the right conditions for a decoding step, then continue
                        continue

                    # RNN
                    startT = time.time()
                    
                    logits, states = self.runSingleDecodingStep(
                        normLayers[input_network_num],
                        inputLayers[input_network_num],
                        rnn=gru_model,
                        x=tf.constant(recent_data, dtype=tf.float32),
                        states=states,
                        applySmooth=True,
                        kernelSD=args['smoothKernelSD'])

                    if verbose:
                        logging.info(F'RNN time: {(time.time()-startT):0.4f} s')

                    # process logits
                    logits.numpy()
                    logits = self.rearrange_speech_logits(logits, has_sil=True)

                    # increment blank or non-blank tick counters
                    if np.argmax(logits, -1) == blankClass:
                        blankTicks += 1
                    else:
                        blankTicks = 0
                        nonBlankTicks += 1

                    self.r.xadd(output_stream, {'logits': logits[0,0,:].tobytes(order='C') })
                    self.r.xadd(self.NAME,
                        {b'logits': logits[0,0,:].tobytes(order='C'),
                         sync_key: sync_dict,
                         time_key: np.uint64(time.monotonic_ns()).tobytes()})

                    if verbose:
                        logging.info(F'Blank ticks: {blankTicks}, non-blank ticks: {nonBlankTicks}')


                    # optionally interpolate logits
                    if logit_interpolation_factor > 1 and prev_logits is not None:
                        combined_logits = np.concatenate((prev_logits, logits), axis=1)
                        prev_logits = logits
                        f = interp1d(np.arange(combined_logits.shape[1]), combined_logits, axis=1)
                        combined_logits_interp = f(np.linspace(0, combined_logits.shape[1]-1, logit_interpolation_factor+1))
                        logits = combined_logits_interp[:, 1:, :]

                    elif logit_interpolation_factor > 1 and prev_logits is None:
                        prev_logits = logits


                    # Put logits into language model
                    startT = time.time()
                    for i in range(logits.shape[1]):
                        if use_local_lm:
                            # get the partially decoded sentence via the local lm
                            lm_decoder.DecodeNumpy(ngramDecoder,
                                                    logits[0,[i],:],
                                                    np.zeros_like(logits[0,[i],:]),
                                                    np.log(blankPenalty))
                            lm_response = ngramDecoder.result()
                            decoded = lm_response[0].sentence

                        else:
                            # get the partially decoded sentence from the remote lm-pc via redis stream
                            self.r.xadd(remote_lm_input_stream, {'logits': np.float32(logits[:,[i],:]).tobytes()})
                            remote_lm_output = self.r.xread({remote_lm_output_partial_stream: remote_lm_output_partial_lastEntrySeen}, block=0, count=1)
                            for entry_id, entry_data in remote_lm_output[0][1]:
                                remote_lm_output_partial_lastEntrySeen = entry_id
                                decoded = entry_data[b'lm_response_partial'].decode()
                        
                    decoded = self.replace_words(decoded, word_substitutions)

                    if verbose:
                        logging.info(F'LM time: {(time.time()-startT):0.4f} s')
                        logging.info(F'Partial decoded sentence: {decoded}')
                        print("\n")

                    p = self.r.pipeline()
                    p.xadd(output_stream, {'partial_decoded_sentence': decoded})
                    p.set('partial_decoded_sentence_current', decoded)
                    p.execute()


                    # end of trial after task ends (user button press), ###or after 1.5 seconds of no new phonemes.
                    if forceStop: # or (blankTicks >= 75 and nonBlankTicks > 0):
                        # set decoder back to idle and mark end in redis stream
                        decoderState = State.DONE

                        p = self.r.pipeline()
                        p.xadd(output_stream, {'end': last_entry_seen})
                        # set the "final" decoded sentence as the last partially decoded one until we have the actual final one.
                        p.set('final_decoded_sentence_current', f'{decoded} ...')
                        p.execute()
                        
                        # reset recent data to zeros
                        recent_data = np.zeros([1, bufferSize, args['dataset']['nInputFeatures']], dtype=np.float32)

                        # reset RNN states
                        states = [gru_model.initStates] + [None] * len(gru_model.rnnLayers)
                        prev_logits = None

                        # reset counter variables
                        frameCounter = 0
                        blankTicks = 0
                        nonBlankTicks = 0

                        # finalize LM
                        startT = time.time()
                        if use_local_lm:
                            # get the final result from the local lm
                            ngramDecoder.FinishDecoding()
                            lm_response = ngramDecoder.result()
                            decoded = lm_response[0].sentence
                        else:
                            # wait to get the final result from the remote lm-pc via redis
                            self.r.set('lm_finalize_flag', 1)
                            remote_lm_output = self.r.xread({remote_lm_output_final_stream: remote_lm_output_final_lastEntrySeen}, block=0, count=1)
                            for entry_id, entry_data in remote_lm_output[0][1]:
                                remote_lm_output_final_lastEntrySeen = entry_id
                                decoded = entry_data[b'lm_response_final'].decode()

                        decoded = self.replace_words(decoded, word_substitutions)

                        if verbose:
                            logging.info(F'LM finishing time: {(time.time()-startT):0.4f} s')
                            logging.info(F'Final decoded sentence: {decoded}')
                            print('')

                        if auto_punctuation:
                            startT = time.time()

                            self.r.xadd('text_for_punctuation', {'text_for_punctuation': decoded})
                            read_result = self.r.xread(
                                {'punctuated_text': self.punctuation_last_input_entry_seen},
                                count = 1,
                                block = 5000 # ms
                            )

                            # timeout if no data received for X ms
                            if len(read_result) == 0:
                                logging.warning('Auto-punctuated text not received within 5 seconds. Skipping punctuation.')
                                self.punctuation_last_input_entry_seen = self.get_current_redis_time_ms()
                            else:
                                # read this data snippet
                                for entry_id, entry_dict in read_result[0][1]:
                                    self.punctuation_last_input_entry_seen = entry_id
                                    decoded = entry_dict[b'punctuated_text'].decode()

                                if verbose:
                                    logging.info(F'Auto-punctuation time: {(time.time()-startT):0.4f} s')
                                    logging.info(F'Punctuated sentence: {decoded}')
                                    print('')

                        # add final decoded sentence to redis, and reset LM
                        p = self.r.pipeline()
                        p.xadd(output_stream, {'final_decoded_sentence': decoded})
                        p.xadd('tts_final_decoded_sentence', {'final_decoded_sentence': decoded}) # this is for the TTS node
                        p.set('final_decoded_sentence_current', decoded)
                        p.set('partial_decoded_sentence_current', '')
                        p.execute()

                        # reset LM
                        if use_local_lm:
                            ngramDecoder.Reset()
                        else:
                            self.reset_remote_lm()
                        if verbose:
                            logging.info(F'Language model reset.')

                        if adaptMean:
                            stats.update(np.stack(sentenceBuffer, 0)) # update normalization statistics
                            self.r.xadd('brainToText_normStats', {'mean': np.array(stats.mean).tobytes(), 'std': np.array(stats.std).tobytes()})
                            logging.info(F'Updated normalization statistics.')

                            if autosaveStats: # save stats
                                os.makedirs(autosaveStats_path, exist_ok=True)
                                logging.info(F'Saving normalization statistics to: {autosaveStats_path}')
                                outDir = autosaveStats_path
                                np.save(os.path.join(outDir, 'blockMean'), np.squeeze(stats.mean))
                                np.save(os.path.join(outDir, 'blockStd'), np.squeeze(stats.std))

                        sentenceBuffer = [] # reset sentence buffer
                        if verbose:
                            logging.info(f'Stop was just forced. Decoder state: {decoderState}')

    def terminate(self, sig, frame):
        logging.info('SIGINT received, Exiting')
        gc.collect()
        sys.exit(0)

if __name__ == "__main__":
    gc.disable()

    node = brainToText_closedLoop()
    node.run()

    gc.collect()