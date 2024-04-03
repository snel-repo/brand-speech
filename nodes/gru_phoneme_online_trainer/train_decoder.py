import tensorflow as tf
import random
from datetime import datetime
import os
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pathlib
import logging
import json
import sys

from speech_utils.models import models
from speech_utils.utils.gaussSmooth import gaussSmooth

from omegaconf import OmegaConf

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class brainToText_decoder(object):
    # This class will initialize and train a brain-to-text phoneme decoder.
    # Written by Nick Card with reference to Stanford NPTL's decoding function.

    # init will create models, load data, and create datasets
    def __init__(self, args):
        self.args = args

        if args['mode']=='train':
            os.makedirs(self.args['outputDir'], exist_ok=False)

        # set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')
        fh = logging.FileHandler(str(pathlib.Path(self.args['outputDir'],'training_log')))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # self.logger.basicConfig(level=logging.INFO,
        #                     handlers=[
        #                         logging.FileHandler(str(pathlib.Path(self.args['outputDir'],'training_log'))),
        #                         logging.StreamHandler(),
        #                         ],
        #                     format="%(asctime)-15s %(message)s")

        # check that the provided session directories actually exist!
        for x in range(len(self.args['dataset']['sessions'])):
            assert os.path.isdir(self.args['dataset']['dataDir'][x]), "Session directory " + self.args['dataset']['dataDir'][x] + " does not exist!"

        # set GPU for tensorflow to use
        self.logger.info(f"Using GPU #: {self.args['gpuNumber']}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args['gpuNumber'])

        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)


        # set seed - either random or specified #
        if self.args['seed']==-1:
            # random seed
            seed = datetime.now().microsecond
        else:
            seed = self.args['seed']
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)


        # initialize GRU
        self.gru_model = models.GRU(self.args['model']['nUnits'], 
                                self.args['model']['weightReg'], 
                                self.args['dataset']['nClasses']+1, 
                                self.args['model']['dropout'], 
                                self.args['model']['nLayers'], 
                                self.args['model']['patch_size'], 
                                self.args['model']['patch_stride'])
        self.gru_model(tf.keras.Input(shape=(None, self.args['model']['inputNetwork']['inputLayerSizes'][-1])))
        self.gru_model.trainable = self.args['model']['trainable']
        self.logger.info(self.gru_model.summary())


        # Build day transformation networks and normalization layers
        self.nInputLayers = len(self.args['dataset']['dataDir'])
        self.inputLayers = []
        self.normLayers = []

        for layerIdx in range(self.nInputLayers):
            normLayer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[self.args['dataset']['nInputFeatures']])

            inputModel = tf.keras.Sequential()
            inputModel.add(tf.keras.Input(shape=(None, self.args['dataset']['nInputFeatures'])))
            inputModel.add(tf.keras.layers.Dense(self.args['model']['inputLayerSize'],
                                                    activation = self.args['model']['inputNetwork']['activation'],
                                                    kernel_initializer = tf.keras.initializers.identity(),
                                                    kernel_regularizer = tf.keras.regularizers.L2(self.args['model']['weightReg'])))
            inputModel.add(tf.keras.layers.Dropout(rate = self.args['model']['inputNetwork']['dropout']))
            inputModel.trainable = self.args['model']['inputNetwork']['trainable']
            inputModel.summary()

            self.inputLayers.append(inputModel)
            self.normLayers.append(normLayer)
        self.logger.info("Initialized decoding model and input networks.")
        

        # create optimizer
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(self.args['learnRateStart'],
                                                                        self.args['nBatchesToTrain'],
                                                                        self.args['learnRateEnd'],
                                                                        self.args['learnRatePower'], cycle=False, name=None)
        self.optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-01, learning_rate=learning_rate_fn)
        self.logger.info("Initialized optimizer.")


        #define a list of all trainable variables for optimization
        self.trainableVariables = []
        if self.args['trainableBackend']==True:
            self.trainableVariables.extend(self.gru_model.trainable_variables)
        if self.args['trainableInput']==True:
            for x in range(len(self.inputLayers)):
                self.trainableVariables.extend(self.inputLayers[x].trainable_variables)
        self.logger.info("Set trainable variables.")

        
        # load data and build datasets
        self.tfTrainDatasets = []
        self.tfValDatasets = []

        tfdata_param_dict = {}

        for i in range(len(self.args['dataset']['sessions'])):
            
            try:
                with open(os.path.join(self.args['dataset']['dataDir'][i], 'mat_to_tfrecord_params.json')) as f:
                    tfdata_param_dict[self.args['dataset']['sessions'][i]] = json.load(f)
            except:
                self.logger.info(f"Could not find mat_to_tfrecord_params.json in {self.args['dataset']['dataDir'][i]}")

            # trainDir = os.path.join(self.args['dataset']['dataDir'][0], self.args['dataset']['sessions'][i], 'train')
            trainDir = os.path.join(self.args['dataset']['dataDir'][i], 'train')
            files = sorted([str(x) for x in pathlib.Path(trainDir).glob("*.tfrecord")])
            trainDataset = tf.data.TFRecordDataset(files)

            self.logger.info("Loaded data from: " + trainDir)

            # this is the encoding for the tfrecord data files
            datasetFeatures = {
                "inputFeatures": tf.io.FixedLenSequenceFeature([self.args['dataset']['nInputFeatures']], tf.float32, allow_missing=True),
                #"classLabelsOneHot": tf.io.FixedLenSequenceFeature([self.nClasses+1], tf.float32, allow_missing=True),
                "newClassSignal": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                "ceMask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                "seqClassIDs": tf.io.FixedLenFeature((self.args['dataset']['maxSeqElements']), tf.int64),
                "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
                "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
                "transcription": tf.io.FixedLenFeature((self.args['dataset']['maxSeqElements']), tf.int64),
                "block_num": tf.io.FixedLenFeature((), tf.int64),
                "trial_num": tf.io.FixedLenFeature((), tf.int64),
            }

            # use tfrecord encoding to parse data into dictionary
            def parseDatasetFunctionSimple(exampleProto):
                return tf.io.parse_single_example(exampleProto, datasetFeatures)
            trainDataset = trainDataset.map(parseDatasetFunctionSimple, num_parallel_calls=tf.data.AUTOTUNE)

            # Shuffle and transform data if training
            trainDataset = trainDataset.shuffle(self.args['dataset']['bufferSize'])
            trainDataset = trainDataset.repeat()
            trainDataset = trainDataset.padded_batch(self.args['batchSize'])
            trainDataset = trainDataset.prefetch(tf.data.AUTOTUNE)


            # validation dataset
            # valDir = os.path.join(self.args['dataset']['dataDir'][0], self.args['dataset']['sessions'][i], 'test')
            valDir = os.path.join(self.args['dataset']['dataDir'][i], 'test')
            files = sorted([str(x) for x in pathlib.Path(valDir).glob("*.tfrecord")])
            valDataset = tf.data.TFRecordDataset(files)   

            # use encoding to parse data into dictionary
            valDataset = valDataset.map(parseDatasetFunctionSimple, num_parallel_calls=tf.data.AUTOTUNE)

            # no shuffling for validation data
            valDataset = valDataset.padded_batch(self.args['batchSize'])
            valDataset = valDataset.prefetch(tf.data.AUTOTUNE)

            # append this day's data to dataset vars
            self.tfTrainDatasets.append(trainDataset)
            self.tfValDatasets.append(valDataset)

            self.logger.info("Loaded data from: " + valDir)


        # create training data iterators and selectors
        self.trainDatasetIterators = [iter(d) for d in self.tfTrainDatasets]
        self.trainDatasetSelector = {}
        for x in range(len(args['dataset']['sessions'])):
            # this selector will automatically put iterated data through the datasetLayerTransform function
            self.trainDatasetSelector[x] = lambda x=x: self.datasetLayerTransform(self.trainDatasetIterators[x].get_next(),
                                                                        self.normLayers[x],
                                                                        args['dataset']['whiteNoiseSD'],
                                                                        args['dataset']['constantOffsetSD'],
                                                                        args['dataset']['randomWalkSD'],
                                                                        args['dataset']['staticGainSD'],
                                                                        args['dataset']['randomCut'])

        self.logger.info("Loaded all data and created datasets and iterators.")


        # optionally load pre-existing model
        if self.args['loadDir'] != None and os.path.exists(os.path.join(self.args['loadDir'], 'checkpoint')):
            ckptVars = {}
            ckptVars['net'] = self.gru_model
            for x in range(len(self.normLayers)):
                ckptVars['normLayer_'+str(x)] = self.normLayers[x]
                ckptVars['inputLayer_'+str(x)] = self.inputLayers[x]

            ckptPath = tf.train.latest_checkpoint(self.args['loadDir'])
            self.logger.info('Loading pre-trained RNN from : ' + str(os.path.abspath(ckptPath)))
            self.checkpoint = tf.train.Checkpoint(**ckptVars)
            self.checkpoint.restore(ckptPath).expect_partial()

            ckptVars['step'] = tf.Variable(0)
            ckptVars['bestValCer'] = tf.Variable(1.0)
            ckptVars['optimizer'] = self.optimizer
            self.checkpoint = tf.train.Checkpoint(**ckptVars)


        if self.args['mode']=='train':
            os.makedirs(self.args['outputDir'], exist_ok=True)
            save_fname = str(pathlib.Path(self.args['outputDir'],'brainToText_decoder_tfdata_params.json'))
            with open(save_fname, 'w') as f:
                json.dump(tfdata_param_dict, f, indent=4, cls=NpEncoder)
            self.logger.info(f"Saved dataset info to: {save_fname}")



    # load a pre-trained model from a specified directory.
    def load_model(self, load_dir):
        ckptPath = tf.train.latest_checkpoint(load_dir)
        ckptVars = {}
        ckptVars['net'] = self.gru_model
        for x in range(len(self.normLayers)):
            ckptVars['normLayer_'+str(x)] = self.normLayers[x]
            ckptVars['inputLayer_'+str(x)] = self.inputLayers[x]
        checkpoint = tf.train.Checkpoint(**ckptVars)
        checkpoint.restore(ckptPath).expect_partial()
        
        self.logger.info("Loaded model from: " + load_dir)

    def _buildOptimizer(self):
        #define the gradient descent optimizer
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(self.args['learnRateStart'],
                                                                            self.args.get('learnRateDecaySteps', self.args['nBatchesToTrain']),
                                                                            end_learning_rate=self.args['learnRateEnd'],
                                                                            power=self.args['learnRatePower'], cycle=False, name=None)

        self.optimizer = tf.keras.optimizers.Adam(
            beta_1=0.9, beta_2=0.999, epsilon=1e-01, learning_rate=learning_rate_fn)

    # this function puts data through normalization layer, adds noise to it, and smooths it
    def datasetLayerTransform(self, dat, normLayer, whiteNoiseSD, constantOffsetSD, randomWalkSD, staticGainSD, randomCut):

        # get neural data and normalize
        features = dat['inputFeatures']
        features = normLayer(dat['inputFeatures'])

        featShape = tf.shape(features)
        batchSize = featShape[0]
        featDim = featShape[2]

        # add static gain noise
        if staticGainSD > 0:
            warpMat = tf.tile(tf.eye(dat['inputFeatures'].shape[2])[tf.newaxis, :, :], [batchSize, 1, 1])
            warpMat += tf.random.normal(tf.shape(warpMat), mean=0, stddev=staticGainSD)
            features = tf.linalg.matmul(features, warpMat)

        # add white noise
        if whiteNoiseSD > 0:
            features += tf.random.normal(featShape, mean=0, stddev=whiteNoiseSD)

        # add constant offset noise
        if constantOffsetSD > 0:
            features += tf.random.normal([batchSize, 1, featDim], mean=0, stddev=constantOffsetSD)

        # add random walk noise
        if randomWalkSD > 0:
            features += tf.math.cumsum(tf.random.normal(
                featShape, mean=0, stddev=randomWalkSD), axis=self.args['randomWalkAxis'])

        # randomly cutoff part of the data timecourse
        if randomCut > 0:
            cut = np.random.randint(0, randomCut)
            features = features[:, cut:, :]
            dat['nTimeSteps'] = dat['nTimeSteps'] - cut

        # gaussian temportal smoothing
        if self.args['smoothInputs']:
            features = gaussSmooth(features, kernelSD=self.args['smoothKernelSD'])

        # construct output dict
        outDict = {'inputFeatures': features,
                    #'classLabelsOneHot': dat['classLabelsOneHot'],
                    'newClassSignal': dat['newClassSignal'],
                    'seqClassIDs': dat['seqClassIDs'],
                    'nTimeSteps': dat['nTimeSteps'],
                    'nSeqElements': dat['nSeqElements'],
                    'ceMask': dat['ceMask'],
                    'transcription': dat['transcription'],
                    'block_num': dat['block_num'],
                    'trial_num': dat['trial_num']}

        return outDict



    # main training loop function
    def train(self):
        self.logger.info("Beginning training for " + str(self.args['nBatchesToTrain']) + " batches.")

        # create variables to track performance
        self.val_per = []
        self.val_per_batch = []
        self.val_per_by_session = []
        self.loss_by_batch = []
        self.gradNorm_by_batch = []

        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        best_val_per = 1.0

        early_stopping = self.args.get('early_stopping', True)
        early_stopping_num_batches = self.args.get('early_stopping_num_batches', 500)
        batches_since_improved = 0
        if early_stopping:
            self.logger.info(f"I will stop training after {self.args['nBatchesToTrain']} batches or " \
                         f"when the validation PER hasn't improved in {early_stopping_num_batches} batches.")
        else:
            self.logger.info(f"I will stop training after {self.args['nBatchesToTrain']} batches.")

        # begin stepping through batches
        trainStart = datetime.now()
        for b in range(self.args['nBatchesToTrain']+1):

            dtStart = datetime.now()

            # choose random day for each batch
            day = np.random.randint(0, len(self.args['dataset']['sessions']))
            # self.logger.info("Day = " + str(day))

            # perform a training step
            trainOut = self.trainStep(tf.constant(day, dtype=tf.int32), tf.constant(day, dtype=tf.int32))

            totalSeconds = (datetime.now()-dtStart).total_seconds()

            self.logger.info(f'Train batch {b}: ' +
                    f'loss: {(trainOut["predictionLoss"] + trainOut["regularizationLoss"]):.2f} ' +
                    f'gradNorm: {trainOut["gradNorm"]:.2f} ' +
                    f'time: {totalSeconds:.3f}')

            self.loss_by_batch.append(trainOut['predictionLoss'] + trainOut['regularizationLoss'])
            self.gradNorm_by_batch.append(trainOut["gradNorm"])


            # Validation
            if b % self.args['batchesPerVal'] == 0:
                dtStart = datetime.now()

                infOut, infOut_day = self.inference()

                totalSeconds = (datetime.now()-dtStart).total_seconds()

                self.logger.info(f'Val batch {b}: ' +
                        f'PER (avg): {infOut["seqErrorRate"]:.3f} ' +
                        f'time: {totalSeconds:.3f}')

                self.val_per.append(infOut["per"])
                self.val_per_batch.append(b)
                self.val_per_by_session.append(infOut_day)

                if infOut['per'] < best_val_per:
                    best_val_per = infOut['per']
                    infOut['best_val_per'] = best_val_per
                    batches_since_improved = 0

                    if save_best_checkpoint:
                        self.save(self.args['outputDir'])

                else:
                    batches_since_improved += self.args['batchesPerVal']

                self.logger.info(f'Batches since validation PER improved: {batches_since_improved}')

                # early stopping
                if early_stopping & (batches_since_improved >= early_stopping_num_batches):
                    self.logger.info(f"Overall validation PER has not improved in " \
                                 f"{early_stopping_num_batches} batches. Stopping training early.")
                    break


        trainTime_s = (datetime.now()-trainStart).total_seconds()
        trainTime_min = trainTime_s / 60

        self.logger.info(f'Best avg PER achieved: {best_val_per:.5f}')
        self.logger.info(f'Total training time: {trainTime_min:.2f} minutes.')

        self.infOut = infOut
        self.train_stats = {}
        self.train_stats['loss_by_batch'] = self.loss_by_batch
        self.train_stats['gradNorm_by_batch'] = self.gradNorm_by_batch
        self.train_stats['val_per'] = self.val_per
        self.train_stats['val_per_batch'] = self.val_per_batch
        self.train_stats['val_per_by_session'] = self.val_per_by_session

        return self.infOut, self.train_stats



    # inference function
    def inference(self, returnData=False):

        infOut = {}
        infOut['logits'] = []
        infOut['logitLengths'] = []
        infOut['decodedSeqs'] = []
        infOut['editDistances'] = []
        infOut['trueSeqLengths'] = []
        infOut['trueSeqs'] = []
        infOut['transcriptions'] = []
        infOut['seqErrorRate'] = []
        infOut['dayVal'] = []
        infOut['session'] = []
        infOut['block_num'] = []
        infOut['trial_num'] = []
        allData = []

        infOut_day = {}

        # step through all validation data for each day
        for day_val in range(0, len(self.args['dataset']['sessions'])):

            if self.args['dataset']['datasetProbabilityVal'][day_val] <= 0:
                continue

            infOut_day[day_val] = {}
            infOut_day[day_val]['logits'] = []
            infOut_day[day_val]['logitLengths'] = []
            infOut_day[day_val]['transcriptions'] = []
            infOut_day[day_val]['decodedSeqs'] = []
            infOut_day[day_val]['trueSeqs'] = []
            infOut_day[day_val]['editDistances'] = []
            infOut_day[day_val]['trueSeqLengths'] = []

            for data in self.tfValDatasets[day_val]:
                
                # perform validation for this chunk of data
                out = self.valStep(data, day_val)

                infOut['logits'].append(out['logits'].numpy())
                infOut['editDistances'].append(out['editDistance'].numpy())
                infOut['trueSeqLengths'].append(out['nSeqElements'].numpy())
                infOut['logitLengths'].append(out['logitLengths'].numpy())
                infOut['trueSeqs'].append(out['trueSeq'].numpy()-1)
                infOut['block_num'].append(out['block_num'].numpy())
                infOut['trial_num'].append(out['trial_num'].numpy())
                infOut['dayVal'].append([day_val] * len(out['trial_num']))
                infOut['session'].append([self.args['dataset']['sessions'][day_val]] * len(out['trial_num']))

                tmp = tf.sparse.to_dense(out['decodedStrings'][0], default_value=-1).numpy()
                paddedMat = np.zeros([tmp.shape[0], self.args['dataset']['maxSeqElements']]).astype(np.int32)-1
                end = min(tmp.shape[1], self.args['dataset']['maxSeqElements'])
                paddedMat[:, :end] = tmp[:, :end]
                infOut['decodedSeqs'].append(paddedMat)

                infOut['transcriptions'].append(out['transcription'].numpy())

                infOut_day[day_val]['logits'].append(out['logits'].numpy())
                infOut_day[day_val]['logitLengths'].append(out['logitLengths'].numpy())
                infOut_day[day_val]['transcriptions'].append(out['transcription'].numpy())
                infOut_day[day_val]['decodedSeqs'].append(paddedMat)
                infOut_day[day_val]['trueSeqs'].append(out['trueSeq'].numpy()-1)
                infOut_day[day_val]['trueSeqLengths'].append(out['nSeqElements'].numpy())
                infOut_day[day_val]['editDistances'].append(out['editDistance'].numpy())

                if returnData:
                    allData.append(data)

            infOut_day[day_val]['logits'] = [l for batch in infOut_day[day_val]['logits'] for l in list(batch)]
            maxLogitLength = max([l.shape[0] for l in infOut_day[day_val]['logits']])
            infOut_day[day_val]['logits'] = [np.pad(l, [[0, maxLogitLength-l.shape[0]], [0, 0]]) for l in infOut_day[day_val]['logits']]
            infOut_day[day_val]['logits'] = np.stack(infOut_day[day_val]['logits'], axis=0)
            infOut_day[day_val]['logitLengths'] = np.concatenate(infOut_day[day_val]['logitLengths'], axis=0)
            infOut_day[day_val]['decodedSeqs'] = np.concatenate(infOut_day[day_val]['decodedSeqs'], axis=0)
            infOut_day[day_val]['editDistances'] = np.concatenate(infOut_day[day_val]['editDistances'], axis=0)
            infOut_day[day_val]['trueSeqLengths'] = np.concatenate(infOut_day[day_val]['trueSeqLengths'], axis=0)
            infOut_day[day_val]['trueSeqs'] = np.concatenate(infOut_day[day_val]['trueSeqs'], axis=0)
            infOut_day[day_val]['transcriptions'] = np.concatenate(infOut_day[day_val]['transcriptions'], axis=0)
            infOut_day[day_val]['per'] = float(np.sum(infOut_day[day_val]['editDistances'])) / np.sum(infOut_day[day_val]['trueSeqLengths'])
            self.logger.info(f'Val batch: ' + f'PER ({self.args["dataset"]["sessions"][day_val]}): {infOut_day[day_val]["per"]:.3f}')

        infOut['logits'] = [l for batch in infOut['logits'] for l in list(batch)]
        maxLogitLength = max([l.shape[0] for l in infOut['logits']])
        infOut['logits'] = [np.pad(l, [[0, maxLogitLength-l.shape[0]], [0, 0]]) for l in infOut['logits']]
        infOut['logits'] = np.stack(infOut['logits'], axis=0)
        infOut['logitLengths'] = np.concatenate(infOut['logitLengths'], axis=0)
        infOut['decodedSeqs'] = np.concatenate(infOut['decodedSeqs'], axis=0)
        infOut['editDistances'] = np.concatenate(infOut['editDistances'], axis=0)
        infOut['trueSeqLengths'] = np.concatenate(infOut['trueSeqLengths'], axis=0)
        infOut['trueSeqs'] = np.concatenate(infOut['trueSeqs'], axis=0)
        infOut['transcriptions'] = np.concatenate(infOut['transcriptions'], axis=0)
        infOut['per'] = np.sum(infOut['editDistances']) / float(np.sum(infOut['trueSeqLengths']))
        infOut['seqErrorRate'] = float(np.sum(infOut['editDistances'])) / np.sum(infOut['trueSeqLengths'])
        infOut['block_num'] = np.concatenate(infOut['block_num'], axis=0)
        infOut['trial_num'] = np.concatenate(infOut['trial_num'], axis=0)
        infOut['dayVal'] = np.concatenate(infOut['dayVal'], axis=0)
        infOut['session'] = np.concatenate(infOut['session'], axis=0)
        
        if returnData:
            return infOut, infOut_day, allData
        else:
            return infOut, infOut_day


    # training step function
    @tf.function()
    def trainStep(self, datasetIdx, layerIdx):
        #loss function & regularization
        data = tf.switch_case(datasetIdx, self.trainDatasetSelector)

        # input network selector for transforming data for each day
        inputTransformSelector = {}
        for x in range(self.nInputLayers):
            inputTransformSelector[x] = lambda x=x: self.inputLayers[x](data['inputFeatures'], training=True)

        # selector for getting the regularization loss for each input network
        regLossSelector = {}
        for x in range(self.nInputLayers):
            regLossSelector[x] = lambda x=x: self.inputLayers[x].losses

        # manual gradient descent implementation
        with tf.GradientTape() as tape:

            # day-specific transform, GRU, and regularization loss
            inputTransformedFeatures = tf.switch_case(layerIdx, inputTransformSelector)
            predictions = self.gru_model(inputTransformedFeatures, training=True)
            regularization_loss = tf.math.add_n(self.gru_model.losses) + \
                tf.math.add_n(tf.switch_case(layerIdx, regLossSelector))

            batchSize = tf.shape(data['inputFeatures'])[0]

            # convert ground truth labels into sparse labels
            sparseLabels = tf.cast(tf.sparse.from_dense(data['seqClassIDs']), dtype=tf.int32)
            sparseLabels = tf.sparse.SparseTensor(
                indices=sparseLabels.indices,
                values=sparseLabels.values-1,
                dense_shape=[batchSize, self.args['dataset']['maxSeqElements']])

            # get number of time steps then adjust for patch size/stride
            nTimeSteps = tf.cast(data['nTimeSteps'] / self.args['model']['subsampleFactor'], dtype=tf.int32)
            kernel = self.args['model']['patch_size']
            stride = self.args['model']['patch_stride']
            nTimeSteps = tf.cast((nTimeSteps - kernel) / stride + 1, dtype=tf.int32)

            # get model prediction loss with CTC
            pred_loss = tf.compat.v1.nn.ctc_loss_v2(sparseLabels,
                                                    predictions,
                                                    None,
                                                    nTimeSteps,
                                                    logits_time_major=False,
                                                    unique=None,
                                                    blank_index=-1,
                                                    name=None,
                                                    # ignore_longer_outputs_than_inputs=True
                                                    )
            pred_loss = tf.reduce_mean(pred_loss)

            total_loss = pred_loss + regularization_loss

        #compute gradients + clip
        grads = tape.gradient(total_loss, self.trainableVariables)
        grads, gradNorm = tf.clip_by_global_norm(grads, self.args['gradClipValue'])

        #only apply if gradients are finite
        allIsFinite = []
        for g in grads:
            if g != None:
                allIsFinite.append(tf.reduce_all(tf.math.is_finite(g)))
        gradIsFinite = tf.reduce_all(tf.stack(allIsFinite))

        # only apply gradients that are not "None" to trainable variables 
        if gradIsFinite:
            self.optimizer.apply_gradients(
                (grd, var) 
                for (grd, var) in zip(grads, self.trainableVariables) 
                if grd is not None
            )

        # construct output dictionary
        output = {}
        output['logits'] = predictions
        #output['rnnUnits'] = intermediate_output
        output['inputFeatures'] = data['inputFeatures']
        output['predictionLoss'] = pred_loss
        output['regularizationLoss'] = regularization_loss
        output['gradNorm'] = gradNorm
        output['seqIDs'] = data['seqClassIDs']
        output['seqErrorRate'] = tf.constant(0.0)

        return output


    # validation function
    def valStep(self, data, day_val):
        # normalize data, add no noise because its validation
        data = self.datasetLayerTransform(data, self.normLayers[day_val], 0, 0, 0, 0, 0)

        # day-specific transform and then predictions
        inputTransformedFeatures = self.inputLayers[day_val](data['inputFeatures'], training=False)
        predictions = self.gru_model(inputTransformedFeatures, training=False)

        batchSize = tf.shape(data['seqClassIDs'])[0]

        # create sparse labels from ground truth labels
        sparseLabels = tf.cast(tf.sparse.from_dense(data['seqClassIDs']), dtype=tf.int32)
        sparseLabels = tf.sparse.SparseTensor(
            indices=sparseLabels.indices,
            values=sparseLabels.values-1,
            dense_shape=[batchSize, self.args['dataset']['maxSeqElements']])

        # time steps adjusted by patch size/stride
        nTimeSteps = tf.cast(data['nTimeSteps'] / self.args['model']['subsampleFactor'], dtype=tf.int32)
        kernel = self.args['model']['patch_size']
        stride = self.args['model']['patch_stride']
        nTimeSteps = tf.cast((nTimeSteps - kernel) / stride + 1, dtype=tf.int32)

        # calculate prediction loss with CTC
        pred_loss = tf.compat.v1.nn.ctc_loss_v2(sparseLabels, predictions, tf.cast(data['nSeqElements'], dtype=tf.int32), nTimeSteps,
                                                logits_time_major=False, unique=None, blank_index=-1, name=None)
        pred_loss = tf.reduce_mean(pred_loss)

        # convert prediction to string, calculate phoneme error rate
        decodedStrings, _ = tf.nn.ctc_greedy_decoder(tf.transpose(predictions, [1, 0, 2]), nTimeSteps, merge_repeated=True)
        editDistance = tf.edit_distance(decodedStrings[0], tf.cast(sparseLabels, tf.int64), normalize=False)
        seqErrorRate = tf.cast(tf.reduce_sum(editDistance), dtype=tf.float32)/tf.cast(tf.reduce_sum(data['nSeqElements']), dtype=tf.float32)

        # construct output dictionary
        output = {}
        output['logits'] = predictions
        output['decodedStrings'] = decodedStrings
        output['seqErrorRate'] = seqErrorRate
        output['editDistance'] = editDistance
        output['trueSeq'] = data['seqClassIDs']
        output['nSeqElements'] = data['nSeqElements']
        output['transcription'] = data['transcription']
        output['logitLengths'] = nTimeSteps
        output['block_num'] = data['block_num']
        output['trial_num'] = data['trial_num']

        return output

    
    # function to save the trained decoder and args to a specified location
    def save(self, save_path=None):
        
        # if save path is not specified, save it to the output directory specified in args
        if save_path==None:
            save_path = self.args['outputDir']

        if save_path[-1] != '/':
            save_path = f'{save_path}/'

        # save models
        ckptVars = {}
        ckptVars['net'] = self.gru_model
        for x in range(len(self.inputLayers)):
            ckptVars['normLayer_'+str(x)] = self.normLayers[x]
            ckptVars['inputLayer_'+str(x)] = self.inputLayers[x]
        ckptVars['optimizer'] = self.optimizer

        os.makedirs(save_path, exist_ok=True)

        checkpoint = tf.train.Checkpoint(**ckptVars)
        checkpoint.save(save_path)
        self.logger.info(f'Model checkpoint saved to: {save_path}')

        # save args file too
        with open(os.path.join(save_path, 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)
        self.logger.info(f'Saved training args.yaml to: {save_path}')