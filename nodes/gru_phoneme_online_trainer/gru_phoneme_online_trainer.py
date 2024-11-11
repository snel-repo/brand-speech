from brand import BRANDNode
import logging
from omegaconf import OmegaConf
import signal
import gc
from glob import glob
import os
import sys
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.training import py_checkpoint_reader
from pathlib import Path
import pickle
import time

from speech_utils.utils.gaussSmooth import gaussSmooth

sys.path.append('../brand-modules/brand-speech/gru_phoneme_online_trainer')
from train_decoder import brainToText_decoder
from online_trainer_helpers import (load_model,
                                    add_input_layers,
                                    data_generator,
                                    load_data,
                                    clean_label,
                                    get_phonemes)

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']


class brainToText_onlineTrainer(BRANDNode):
    def __init__(self):
        super().__init__()

        ## Load parameters, using `self.parameters`.
        metadata_stream_name = self.parameters.get("metadata_stream", 'block_metadata')
        metadata_stream = self.r.xrevrange(metadata_stream_name, count=1)
        participant = metadata_stream[0][1].get(b'participant', b'unknown_participant').decode()
        session_name = metadata_stream[0][1].get(b'session_name', b'unknown_session_name').decode()
        
        # CONFIG PATH ----------------------------------------------------------
        self.config_file_path = self.parameters.get("config_file_path", None)
        if self.config_file_path is None:
            self.config_file_path = f'/samba/data/{participant}/{session_name}/RawData/Models/gru_decoder/online_trainer_config.yaml'
            logging.warning(f'No config file path provided. Using default: {self.config_file_path}')
        # RNN INIT PATH --------------------------------------------------------
        self.init_model_dir = self.parameters.get("init_model_dir", None)
        if self.init_model_dir is None:
            self.init_model_dir = f'/samba/data/{participant}/{session_name}/RawData/Models/gru_decoder'
            logging.warning(f'No initial model directory provided. Using default: {self.init_model_dir}')
        
        self.init_model_number = int(self.parameters.get("init_model_number", -1))  # -1 means that the latest model will be used
        # TRAINING PARAMETERS --------------------------------------------------
        self.training_frequency = int(self.parameters.get("training_frequency", 1))  # How often to train the model (every N trials)     
        self.min_num_trials = int(self.parameters.get('min_num_trials', 10))  
        self.max_trial_len_s = int(self.parameters.get('max_trial_len_s', 50))
        self.trial_info_stream = self.parameters.get('trial_info_stream', 'trial_info')
        self.neural_data_stream = self.parameters.get('neural_data_stream', 'binnedFeatures_20ms')  
        self.use_threshold_crossings = bool(self.parameters.get('use_threshold_crossings', True))
        self.use_spike_band_power = bool(self.parameters.get('use_spike_band_power', True))
        self.delay_as_sil = bool(self.parameters.get('delay_as_sil', False))
        gpu_number = str(self.parameters.get("gpu_number", "1"))       # GPU for tensorflow to use. -1 means that GPU is hidden and inference will happen on CPU.
        
        self.verbose = bool(self.parameters.get('verbose', False))

        self.n_features = int(self.parameters.get("n_features", 512))
        self.excl_chans = self.parameters.get("excl_chans", [])
        self.ch_mask_stream = self.parameters.get("ch_mask_stream", None)
        self.tot_ch = int(self.parameters.get("tot_ch", 256))
        self.zero_masked_chans = bool(self.parameters.get("zero_masked_chans", True))

        self.config = OmegaConf.load(self.config_file_path)

        # --------------------------- load channel mask --------------------------------------------
        self.load_channel_mask()

        # ------------------------------ find RNN model path ---------------------------------------
        if not os.path.exists(self.init_model_dir):
            logging.error(f'Initial RNN directory not found at: {self.init_model_dir}')

        if self.init_model_number != -1:
            # choose a specific model number
            model_list, model_nums = self.sort_models([str(x) for x in Path(self.init_model_dir).glob(f'rnn_model_{self.init_model_number}')])
        else:
            # choose the latest model number
            model_list = [str(x) for x in Path(self.init_model_dir).glob(f'rnn_model_*')]
            model_list = [x for x in model_list if 'seed' not in x]
            model_list, model_nums = self.sort_models(model_list)

        if len(model_list) == 0:
            logging.error(f'Initial RNN model #{self.init_model_number} not found at: {self.init_model_dir}')
            
        # set output dir and init model dir
        self.output_dir = os.path.join(self.init_model_dir, f'online_training_rnn_model_{model_nums[-1]}')
        self.init_model_dir = model_list[-1]
        logging.info(f'Using initial model: {self.init_model_dir}')

        # change args in config
        self.config['output_dir'] = str(self.output_dir)
        self.config['init_model_dir'] = str(self.init_model_dir)

        # save config to output path
        os.makedirs(self.config['output_dir'], exist_ok=True)
        with open(os.path.join(self.config.output_dir, 'config_args.yaml'), 'w+') as f:
            OmegaConf.save(config=self.config, f=f)

        self.trialInfo_lastEntrySeen = '0'
        self.token_def = PHONE_DEF_SIL


        # set GPU for tensorflow to use
        logging.info(f"Using GPU #: {gpu_number}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # Load pretrained model
        self.nsd = load_model(self.config['init_model_dir'], self.config['init_model_ckpt_idx'], gpu_number)

        # Add input layers if needed
        add_input_layers(self.nsd,
                         list(range(len(self.nsd.args['dataset']['dataDir']))), #self.nsd.args['dataset']['datasetToLayerMap'],
                         self.config['session_input_layers'])

        # Init optimizer
        self.nsd.args['learnRateStart'] = self.config.learning_rate
        self.nsd.args['learnRateEnd'] = self.config.learning_rate
        self.nsd._buildOptimizer()

        # Restore weights from previous checkpoint if it exists
        ckpt_vars = {
            'net': self.nsd.gru_model,
            'optimizer': self.nsd.optimizer,
        }

        for i, l in enumerate(self.nsd.inputLayers):
            ckpt_vars[f'inputLayer_{i}'] = l

        self.ckpt = tf.train.Checkpoint(**ckpt_vars)

        if Path(self.config.output_dir, 'checkpoint').exists():
            logging.info('Restoring from checkpoint...')
            latest_ckpt = tf.train.latest_checkpoint(self.config.output_dir)
            self.ckpt.restore(latest_ckpt)
            logging.info(f'Restored from checkpoint: {latest_ckpt}')

            # Copy new input layers weights
            reader = py_checkpoint_reader.NewCheckpointReader(latest_ckpt)
            var_to_shape_map = reader.get_variable_to_shape_map()
            ckpt_input_layers = set()
            for layer in var_to_shape_map:
                if 'inputLayer_' in layer:
                    ckpt_input_layers.add(int(layer.split('/')[0].split('_')[-1]))
            logging.info(f'Found {len(ckpt_input_layers)} input layers in checkpoint')

            for i in range(len(self.nsd.inputLayers)):
                if i not in ckpt_input_layers:
                    logging.info(f'Copy input layer {i} weights from {i - 1}')
                    from_layer = self.nsd.inputLayers[i - 1]
                    to_layer = self.nsd.inputLayers[i]
                    for vf, vt in zip(from_layer.variables, to_layer.variables):
                        vt.assign(vf)

        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            self.config.output_dir,
            max_to_keep=5)
        self.ckpt_manager.save()
        logging.info(f'Saved new checkpoint: {self.ckpt_manager.latest_checkpoint}')

        self.update_model_path()

        #  ---------------------------- Load training state -----------------------------
        output_path = Path(self.config.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        self.state = {
            'input_features': [],
            'normalized_input_features': [],
            'labels': [],
            'labels_phonemes': [],
            'transcriptions': [],
            'seq_class_ids': [],
            'losses': [],
            'block_num': [],
        }

        today_date = datetime.date.today()
        today_date = today_date.strftime('%Y.%m.%d')
  
        self.state_path = Path(self.config.output_dir, f'state_{today_date}.pkl')
        self.load_state()


        # ----------- get block number from metadata -------------- 
        metadata_stream = self.r.xread(
            {"block_metadata": 0},
            count=None,
            block=0,    # block until a message is available
        )
        if metadata_stream == []:
            logging.error('No metadata found in redis stream')
            self.block_num = -1
        else:
            self.block_num = int(metadata_stream[0][1][0][1][b'block_number'].decode())

        
        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)


    def sort_models(self, model_list):
        # function to sort params list in ascending order of block number

        model_num = []
        for i in range(len(model_list)):
            model_num.append(int(model_list[i].split('rnn_model_')[1]))

        ind = np.argsort(np.array(model_num))
        model_list = list(np.array(model_list)[ind])
        model_nums = list(np.array(model_num)[ind])

        return model_list, model_nums
    

    def update_model_path(self):
        if os.path.exists(os.path.join(self.config.output_dir, 'checkpoint')):
            ckpt_path = os.path.join(self.config.output_dir, self.ckpt_manager.latest_checkpoint)
            self.r.xadd('decoderControl:stream', {'newModelPath': ckpt_path})


    def load_state(self):
        if self.state_path.exists():
            with self.state_path.open('rb') as f:
                self.state = pickle.load(f)
            logging.info(f'state pickle file found, loading state from {self.state_path}')
        else:
            logging.info(f'No state pickle file found, starting with empty state')


    def save_state(self):
        with self.state_path.open('wb') as f:
            pickle.dump(self.state, f)
    
    
    def get_trial_data_from_redis(self):

        # Get start and end message ids from decoder
        if self.verbose:
            logging.info('Waiting for decoder to run')

        while True:
            # get data from up to 10 new trials
            stream_entry = self.r.xread({self.trial_info_stream: self.trialInfo_lastEntrySeen}, count=10, block=1000)

            # if no entries, try again
            if stream_entry == []:
                continue

            neural_features = []
            cue = []
            trial_paused = []
            trial_timed_out = []

            for entry_id, entry_dict in stream_entry[0][1]:
                self.trialInfo_lastEntrySeen = entry_id

                if self.delay_as_sil:
                    start = np.frombuffer(entry_dict[b'trial_start_redis_time'], dtype=np.uint64)[0]
                else:
                    start = np.frombuffer(entry_dict[b'go_cue_redis_time'], dtype=np.uint64)[0]
                end = np.frombuffer(entry_dict[b'trial_end_redis_time'], dtype=np.uint64)[0]

                cue.append(entry_dict[b'cue'].decode())
                trial_paused.append(int(entry_dict[b'ended_with_pause'].decode()))
                trial_timed_out.append(int(entry_dict[b'ended_with_timeout'].decode()))

                logging.info(f'Got trial info from decoder - cue: {cue[-1]}')

                start = bytes(str(start), 'utf-8')
                end = bytes(str(end), 'utf-8')

                neural_stream = self.r.xrange(self.neural_data_stream, start, end)
                if self.verbose:
                    logging.info(f'neural stream len: {len(neural_stream)}')

                new_neural_features = np.zeros((len(neural_stream), self.nsd.args['dataset']['nInputFeatures']))

                for i, (entry_id, entry_dict) in enumerate(neural_stream):
                    features = np.frombuffer(entry_dict[b'samples'], dtype=np.float32)

                    if self.use_threshold_crossings:
                        threshold_crossings = features[:self.nsd.args['dataset']['nInputFeatures']//2]

                    if self.use_spike_band_power:
                        # quick hack to get mean SBP instead of summed
                        spike_band_power = features[self.nsd.args['dataset']['nInputFeatures']//2:] / 20

                    if self.use_threshold_crossings and self.use_spike_band_power:
                        new_neural_features[i,:] = np.concatenate((threshold_crossings, spike_band_power), axis=0)
                    elif self.use_threshold_crossings:
                        new_neural_features[i,:] = threshold_crossings
                    elif self.use_spike_band_power:
                        new_neural_features[i,:] = spike_band_power

                neural_features.append(new_neural_features)

            return neural_features, cue, trial_paused, trial_timed_out

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
        # Load previous sessions' data
        prev_data_buffer = {}
        for i, sess in enumerate(self.config.sessions):
            prev_data_buffer[self.config.session_input_layers[i]] = load_data(self.nsd.args, self.config.data_dir[i], normalize=False)

        num_trials = 0

        logging.info('Starting online training!')
        while True:

            # Get neural data and label from redis
            neural_data_list, label_list, trial_paused_list, trial_timed_out_list = self.get_trial_data_from_redis()

            # Add data to state
            num_new_good_trials = 0
            for neural_data, label, trial_paused, trial_timed_out in zip(neural_data_list, label_list, trial_paused_list, trial_timed_out_list):
                # skip trial if label is blank, or just "a" or "i"
                if (len(label.split()) == 0) or (label.strip() in ['a','i']):
                    logging.info(f'Label is blank. Skipping sentence.')
                    num_trials += 1
                    continue

                # Clean label and get phonemes
                label = clean_label(label, self.config.task)
                label_phonemes = get_phonemes(label, prepend_sil=self.delay_as_sil)
                if self.verbose:
                    logging.info(f'Cleaned label: {label}')
                    logging.info(f'Phonemes: {label_phonemes}')

                if neural_data.shape[0] < (len(label_phonemes)*4 + 10):
                    logging.info(f'Neural data too short for label. Skipping sentence.')
                    num_trials += 1
                    continue
                elif neural_data.shape[0] > (self.max_trial_len_s*1000 / 20):
                    logging.info(f'Trial was too long (>{self.max_trial_len_s}s). Skipping sentence.')
                    num_trials += 1
                    continue
                elif trial_paused == 1:
                    logging.info(f'Trial paused by CNRA. Skipping sentence.')
                    num_trials += 1
                    continue
                elif trial_timed_out == 1:
                    logging.info(f'Trial timed out. Skipping sentence.')
                    num_trials += 1
                    continue

                # Make transcription and seq_class_ids
                transcription = np.zeros([500], dtype=np.int64)
                transcription[0:len(label)] = np.array([ord(char) for char in label])
                seq_class_ids = np.zeros([500], dtype=np.int64)
                seq_class_ids[0:len(label_phonemes)] = [PHONE_DEF_SIL.index(p) + 1 for p in label_phonemes]

                # Add data to state
                self.state['block_num'].append(self.block_num)
                self.state['input_features'].append(neural_data)
                self.state['labels'].append(label)
                self.state['labels_phonemes'].append(label_phonemes)
                self.state['transcriptions'].append(transcription)
                self.state['seq_class_ids'].append(seq_class_ids)

                if self.verbose:
                    logging.info(f'Current # of sentences: {len(self.state["input_features"])}')

                # Normalize input features
                norm_features = np.concatenate(self.state['input_features'][-self.config.num_norm_sentences:], axis=0)
                mean = np.mean(norm_features, axis=0, keepdims=True)
                std = np.std(norm_features, axis=0, keepdims=True) + 1e-8
                neural_data_norm = (neural_data - mean) / std
                # zero out masked channels
                if self.zero_masked_chans:
                    neural_data_norm_zeroed = np.zeros_like(neural_data_norm)
                    neural_data_norm_zeroed[:, self.ch_mask] = neural_data_norm[:, self.ch_mask]
                    neural_data_norm = neural_data_norm_zeroed
                else:
                    neural_data_norm = neural_data_norm[:, self.ch_mask]
                self.state['normalized_input_features'].append(neural_data_norm)

                num_trials += 1
                num_new_good_trials += 1


            # only proceed to training if we got good new data
            if num_new_good_trials == 0:
                continue

            # only train every N trials
            if (len(self.state['input_features']) >= self.min_num_trials) and (num_trials % self.training_frequency == 0):

                # Prepare data for training
                curr_data_buffer = []
                for feat, label, label_phonemes, transcription, seq_class_ids in zip(self.state['normalized_input_features'],
                                                                                    self.state['labels'], 
                                                                                    self.state['labels_phonemes'], 
                                                                                    self.state['transcriptions'], 
                                                                                    self.state['seq_class_ids']):

                    data = {
                        'inputFeatures': feat,
                        'seqClassIDs': seq_class_ids,
                        'nTimeSteps': feat.shape[0],
                        'nSeqElements': len(label_phonemes),
                        'transcription': transcription
                    }
                    curr_data_buffer.append(data)

                logging.info(f'Starting training. Previous data buffer: {sum([len(d) for d in prev_data_buffer.values()])}, current data buffer: {len(curr_data_buffer)}')
                avg_loss = self.train_single_sentence(
                    prev_data_buffer,
                    curr_data_buffer
                )

                self.state['losses'].append(avg_loss)
                self.save_state()

                self.ckpt_manager.save()
                logging.info(f'Saved new checkpoint: {self.ckpt_manager.latest_checkpoint}')
                self.update_model_path()


            elif len(self.state['input_features']) < self.min_num_trials:
                logging.info(f'Not enough trials to train yet. Currently have {len(self.state["input_features"])}/{self.min_num_trials} trials.')
                self.state['losses'].append(np.nan)
                self.save_state()


            else:
                logging.info(f'Skipping training this trial because I was told to only train every {self.training_frequency} trials.')
                self.state['losses'].append(np.nan)
                self.save_state()

        
    def train_single_sentence(self, prev_data_buffer, curr_data_buffer):
        # Compose datasets
        # logging.info(f'prev_data_buffer: {sum([len(d) for d in prev_data_buffer.values()])}')
        # logging.info(f'curr_data_buffer: {len(curr_data_buffer)}')
        prev_generator = data_generator(prev_data_buffer)
        curr_generator = data_generator({self.config.session_input_layers[-1]: curr_data_buffer})
        output_signature = {
            'layerIdx': tf.TensorSpec(shape=(), dtype=tf.int32),
            'inputFeatures': tf.TensorSpec(shape=(None,  self.nsd.args['dataset']['nInputFeatures']), dtype=tf.float32),
            'seqClassIDs': tf.TensorSpec(shape=(self.nsd.args['dataset']['maxSeqElements']), dtype=tf.int64),
            'nTimeSteps': tf.TensorSpec(shape=(), dtype=tf.int64),
            'nSeqElements': tf.TensorSpec(shape=(), dtype=tf.int64),
            'transcription': tf.TensorSpec(shape=(self.nsd.args['dataset']['maxSeqElements']), dtype=tf.int64)
        }
        prev_dataset = tf.data.Dataset.from_generator(lambda: prev_generator,
                                                      output_signature=output_signature)
        curr_dataset = tf.data.Dataset.from_generator(lambda: curr_generator,
                                                      output_signature=output_signature)
        prev_dataset = prev_dataset.cache().repeat().shuffle(buffer_size=100)
        curr_dataset = curr_dataset.cache().repeat().shuffle(buffer_size=100)
        dataset = tf.data.Dataset.sample_from_datasets(
            [prev_dataset, curr_dataset],
            weights=[1.0 - self.config.new_data_percent, self.config.new_data_percent]
        )
        #dataset = curr_dataset
        dataset = dataset.padded_batch(self.config.batch_size)

        # Train model
        steps = 0
        losses = []
        start_time = time.time()

        for data in dataset:
            loop_start_time = time.time()

            if steps > self.config.min_train_steps and steps > self.config.max_train_steps:
                break

            # if self.config.time_warp_factor > 0:
            #     data = timeWarpDataElement(data, self.config.time_warp_factor)

            try:
                ctc_loss, reg_loss, total_loss, grad_norm = self.train_step(data['inputFeatures'],
                                                                 data['layerIdx'],
                                                                 data['seqClassIDs'],
                                                                 data['nTimeSteps'],
                                                                 self.config.white_noise_sd,
                                                                 self.config.constant_offset_sd,
                                                                 self.config.random_walk_sd,
                                                                 self.config.random_walk_axis
                                                                 )

                if self.verbose:
                    logging.info(
                        f'Step: {steps}, ' +
                        f'CTC loss: {ctc_loss.numpy():.4f}, ' +
                        f'Reg loss: {reg_loss.numpy():.4f}, ' +
                        f'Total loss: {total_loss.numpy():.4f}, ' +
                        f'LR: {self.nsd.optimizer._decayed_lr(tf.float32).numpy():.4f}, ' +
                        f'Grad norm: {grad_norm.numpy():.4f}, ' +
                        f'Time: {time.time() - loop_start_time:.4f}')
                losses.append(total_loss.numpy())
                steps += 1

                if np.mean(losses[-10:]) < self.config.loss_threshold and steps > self.config.min_train_steps:
                    break
            except tf.errors.InvalidArgumentError as e:
                logging.error(f'Invalid argument error: {e}')

        logging.info(
            f'Online training summary: ' +
            f'Steps: {steps}, ' +
            f'CTC loss: {ctc_loss.numpy():.4f}, ' +
            f'Reg loss: {reg_loss.numpy():.4f}, ' +
            f'Total loss: {total_loss.numpy():.4f}, ' +
            f'LR: {self.nsd.optimizer._decayed_lr(tf.float32).numpy():.4f}, ' +
            f'Grad norm: {grad_norm.numpy():.4f}, ' +
            f'Time: {time.time() - start_time:.4f}')

        return np.mean(losses)


    def train_step(self,
                   inputs,
                   layerIdx,
                   labels,
                   time_steps,
                   white_noise_sd,
                   constant_offset_sd,
                   random_walk_sd,
                   random_walk_axis,
                   max_seq_len=500,
                   grad_clip_value=10.0):
        input_shape = tf.shape(inputs)
        B = input_shape[0]
        C = input_shape[2]

        # Add noise
        inputs += tf.random.normal(shape=input_shape, mean=0, stddev=white_noise_sd)
        inputs += tf.random.normal([B, 1, C], mean=0, stddev=constant_offset_sd)
        inputs += tf.math.cumsum(tf.random.normal(shape=input_shape, mean=0, stddev=random_walk_sd), axis=random_walk_axis)
        
        # temporal smoothing
        inputs = gaussSmooth(inputs, kernelSD=self.nsd.args['smoothKernelSD'])

        # Compute loss
        with tf.GradientTape() as tape:
            new_inputs = []
            for i in range(B):
                new_inputs.append(self.nsd.inputLayers[layerIdx[i]](inputs[i:i+1]))
            new_inputs = tf.concat(new_inputs, axis=0)
            logits = self.nsd.gru_model(new_inputs, training=True)

            sparse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int32)
            sparse_labels = tf.sparse.SparseTensor(
                indices=sparse_labels.indices,
                values=sparse_labels.values-1,
                dense_shape=[1, max_seq_len])
            
            # time_steps = self.nsd.gru_model.getSubsampledTimeSteps(time_steps)
            kernel = self.nsd.args['model']['patch_size']
            stride = self.nsd.args['model']['patch_stride']
            time_steps = tf.cast((time_steps - kernel) / stride + 1, dtype=tf.int32)

            ctc_loss = tf.compat.v1.nn.ctc_loss_v2(sparse_labels,
                                                   logits,
                                                   None,
                                                   time_steps,
                                                   logits_time_major=False,
                                                   unique=None,
                                                   blank_index=-1,
                                                   name=None)
            ctc_loss = tf.reduce_mean(ctc_loss)
            reg_loss = tf.math.add_n(
                self.nsd.gru_model.losses) + \
                tf.math.add_n(self.nsd.inputLayers[self.config.train_layer_idx].losses)  # TODO: Always assume last layer is for training?
            total_loss = ctc_loss + reg_loss

        # Apply gradients
        trainables = self.nsd.gru_model.trainable_variables + \
            self.nsd.inputLayers[self.config.train_layer_idx].trainable_variables
        grads = tape.gradient(total_loss, trainables)
        grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip_value)
        self.nsd.optimizer.apply_gradients(zip(grads, trainables))

        return ctc_loss, reg_loss, total_loss, grad_norm
    

    def terminate(self, sig, frame):
        logging.info('SIGINT received, saving current state pickle file then exiting')
        self.save_state()
        gc.collect()
        sys.exit(0)

if __name__ == "__main__":
    gc.disable()

    node = brainToText_onlineTrainer()
    node.run()

    gc.collect()