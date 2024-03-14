# rdb_to_mat.py
# Nick Card, July 2023
# Modified by Sam Nason-Tomaszewski, March 2024
# This script formats and converts a .rdb file to a .mat file. Designed for use with brainToText.
# The .mat file will be saved to the same directory as the supergraph is est to save to.

import argparse
import datetime
import json
import logging
import numpy as np
import os
import scipy.io
import signal
import matplotlib.pyplot as plt
import sys

from brand.redis import RedisLoggingHandler

from pathlib import Path

from redis import Redis

###############################################
# Initialize script
###############################################

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--nickname', type=str, required=True)
ap.add_argument('-i', '--redis_host', type=str, required=True)
ap.add_argument('-p', '--redis_port', type=int, required=True)
ap.add_argument('-s', '--redis_socket', type=str, required=False)
args = ap.parse_args()

NAME = args.nickname
redis_host = args.redis_host
redis_port = args.redis_port
redis_socket = args.redis_socket

loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{NAME}] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)

def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(1)

# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)

###############################################
# Connect to redis
###############################################
try:
    logging.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    r = Redis(redis_host, redis_port, redis_socket, retry_on_timeout=True)
    r.ping()
except ConnectionError as e:
    logging.error(f"Error with Redis connection, check again: {e}")
    sys.exit(1)
except:
    logging.error('Failed to connect to Redis. Exiting.')
    sys.exit(1)

logging.getLogger().addHandler(RedisLoggingHandler(r, NAME))

logging.info('Redis connection successful.')

###############################################
# Load all stream and NWB info
# do outside of main loop for eventual translation to real-time
###############################################

try:
    model_stream_entry = r.xrevrange(b'supergraph_stream', '+', '-', 1)[0]
except IndexError as e:
    logging.error(
        f"No model published to supergraph_stream in Redis. Exiting.")
    sys.exit(1)

entry_id, entry_dict = model_stream_entry
model_data = json.loads(entry_dict[b'data'].decode())

if NAME in model_data['derivatives']:
    graph_params = model_data['derivatives'][NAME]['parameters']
else:
    logging.warning(
        f"No {NAME} derivative configuration in the current graph. Exiting.")
    sys.exit(0)

# Get parameters
bin_size = graph_params.setdefault('bin_size', 20)
save_location = graph_params.setdefault('save_location')
include_audio = graph_params.setdefault('include_audio', 1)
metadata_stream_name = graph_params.setdefault('metadata_stream', 'block_metadata')
trial_info_stream_name = graph_params.setdefault('trial_info_stream', 'trial_info')
raw_neural_stream_name = graph_params.setdefault('raw_neural_stream', 'nsp_neural')
binned_neural_stream_name = graph_params.setdefault('binned_neural_stream', 'binned_spikes')
analog_stream_name = graph_params.setdefault('analog_stream', 'nsp_analog')
decoder_output_stream_name = graph_params.setdefault('decoder_output_stream', 'binned:decoderOutput:stream')
norm_stats_stream_name = graph_params.setdefault('norm_stats_stream', 'brainToText_normStats')

# ---------------------------------------------------------------------------------------------------------------

# get info from metadata stream
metadata_stream = r.xrevrange(metadata_stream_name, count=1)

participant = metadata_stream[0][1].get(b'participant', b'unknown_participant').decode()
session_name = metadata_stream[0][1].get(b'session_name', b'unknown_session_name').decode()
session_description = metadata_stream[0][1].get(b'session_description', b'').decode()
block_decription = metadata_stream[0][1].get(b'block_description', b'').decode()
graph_name = model_data['graph_name']
block_num = int(metadata_stream[0][1].get(b'block_number', b'-1').decode())
start_time = np.frombuffer(metadata_stream[0][1].get(b'start_time', b'-1'), dtype=np.uint64)[0]

# ---------------------------------------------------------------------------------------------------------------

# initialize trial info stream
trial_info_stream = r.xrange(trial_info_stream_name)

# extract trial info to variables
cue = []
delay_duration = []
inter_trial_duration = []
trial_start_redis_time = []
trial_start_nsp_neural_time = []
trial_start_nsp_analog_time = []
go_cue_redis_time = []
go_cue_nsp_neural_time = []
go_cue_nsp_analog_time = []
trial_end_redis_time = []
trial_end_nsp_neural_time = []
trial_end_nsp_analog_time = []
trial_paused_by_CNRA = []
trial_timed_out = []
trial_accuracy_confirmation = []

for i in range(len(trial_info_stream)):
    cue.append(trial_info_stream[i][1][b'cue'].decode())
    delay_duration.append(int(trial_info_stream[i][1][b'delay_duration'].decode()))
    inter_trial_duration.append(int(trial_info_stream[i][1][b'inter_trial_duration'].decode()))

    trial_start_redis_time.append( (np.frombuffer(trial_info_stream[i][1][b'trial_start_redis_time'], dtype=np.uint64)[0]).astype(np.float64) )
    trial_start_nsp_neural_time.append( (np.frombuffer(trial_info_stream[i][1][b'trial_start_nsp_neural_time'], dtype=np.uint64)[0]).astype(np.float64) )
    trial_start_nsp_analog_time.append( (np.frombuffer(trial_info_stream[i][1][b'trial_start_nsp_analog_time'], dtype=np.uint64)[0]).astype(np.float64) )

    go_cue_redis_time.append( (np.frombuffer(trial_info_stream[i][1][b'go_cue_redis_time'], dtype=np.uint64)[0]).astype(np.float64) )
    go_cue_nsp_neural_time.append( (np.frombuffer(trial_info_stream[i][1][b'go_cue_nsp_neural_time'], dtype=np.uint64)[0]).astype(np.float64) )
    go_cue_nsp_analog_time.append( (np.frombuffer(trial_info_stream[i][1][b'go_cue_nsp_analog_time'], dtype=np.uint64)[0]).astype(np.float64) )

    trial_end_redis_time.append( (np.frombuffer(trial_info_stream[i][1][b'trial_end_redis_time'], dtype=np.uint64)[0]).astype(np.float64) )
    trial_end_nsp_neural_time.append( (np.frombuffer(trial_info_stream[i][1][b'trial_end_nsp_neural_time'], dtype=np.uint64)[0]).astype(np.float64) )
    trial_end_nsp_analog_time.append( (np.frombuffer(trial_info_stream[i][1][b'trial_end_nsp_analog_time'], dtype=np.uint64)[0]).astype(np.float64) )

    trial_paused_by_CNRA.append( int(trial_info_stream[i][1][b'ended_with_pause'].decode()) )
    if trial_paused_by_CNRA[-1] == 1:
        logging.info(f'Trial #{i+1} ended by being paused by CNRA. Cue: {cue[-1]}')

    trial_timed_out.append( int(trial_info_stream[i][1][b'ended_with_timeout'].decode()) )
    if trial_timed_out[-1] == 1:
        logging.info(f'Trial #{i+1} timed out. Cue: {cue[-1]}')

    # get decoding accuracy confirmation. Only present for personal use blocks. If not present, put -1 as a placeholder.
    if b'decoded_correctly' in trial_info_stream[i][1]:
        trial_accuracy_confirmation.append(int(trial_info_stream[i][1][b'decoded_correctly'].decode()))
    else:
        trial_accuracy_confirmation.append(int(-1))

logging.info(f'All trials that were paused: {[i for i,v in enumerate(trial_paused_by_CNRA) if v==1]}')
logging.info(f'All trials that timed out: {[i for i,v in enumerate(trial_timed_out) if v==1]}')
# ---------------------------------------------------------------------------------------------------------------

# initialize the binnedFeatures stream, assuming that it exists
stream = r.xrange(binned_neural_stream_name)
first_neural_data_point = np.frombuffer(stream[0][1][b'samples'], dtype=np.float32)
n_channels = int(len(first_neural_data_point) / 2)

# need to load relation between nsp_idx_1 and NSP timestamps
# load in all nsp_neural entries
nsp_neural = r.xrange(raw_neural_stream_name)
nsp_neural_tracking_ids = np.array([np.frombuffer(entry[b'tracking_id'], dtype=np.uint64) for _, entry in nsp_neural]).squeeze()
nsp_neural_timestamps = np.array([np.frombuffer(entry[b'timestamps'], dtype=np.uint64)[-1] for _, entry in nsp_neural]).squeeze()
del nsp_neural

# extract pertinent parts of the binnedFeatures stream to variables
binned_neural_threshold_crossings = np.zeros([len(stream), n_channels], dtype=np.int16)
binned_neural_spike_band_power = np.zeros([len(stream), n_channels], dtype=np.float32)
binned_neural_nsp_timestamp = np.zeros(len(stream), dtype=np.float64)
binned_neural_redis_clock = np.zeros(len(stream), dtype=np.uint64)

for x in range(len(stream)):
    samples = np.frombuffer(stream[x][1][b'samples'], dtype=np.float32)
    binned_neural_threshold_crossings[x,:] = samples[:n_channels].astype(np.int16)
    binned_neural_spike_band_power[x,:] = samples[n_channels:] / bin_size # was summed across bins during collection
    sync = json.loads(stream[x][1][b'sync'])
    neural_tracking_id_idx = np.argwhere(nsp_neural_tracking_ids == sync['nsp_idx_1']).item()
    binned_neural_nsp_timestamp[x] = nsp_neural_timestamps[neural_tracking_id_idx].astype(np.float64)
    binned_neural_redis_clock_ = str(stream[x][0].decode()).split('-') #get rid of the hyphenated part of the redis timestammp
    binned_neural_redis_clock[x] = int(binned_neural_redis_clock_[0])

# trim binned neural data to begin at the start of the first trial, end at the end of the last trial
nsp_neural_start_time = trial_start_nsp_neural_time[0]
nsp_neural_end_time = trial_end_nsp_neural_time[-1]

binned_neural_start_ind = np.argmin(np.abs(binned_neural_nsp_timestamp - nsp_neural_start_time))
binned_neural_end_ind = np.argmin(np.abs(binned_neural_nsp_timestamp - nsp_neural_end_time))

binned_neural_threshold_crossings = binned_neural_threshold_crossings[binned_neural_start_ind:binned_neural_end_ind]
binned_neural_spike_band_power = binned_neural_spike_band_power[binned_neural_start_ind:binned_neural_end_ind]
binned_neural_nsp_timestamp = binned_neural_nsp_timestamp[binned_neural_start_ind:binned_neural_end_ind]
binned_neural_redis_clock = binned_neural_redis_clock[binned_neural_start_ind:binned_neural_end_ind]


# trim binned neural data for each trial (GO to trial end), bin to 20ms, and then get normalization statistics.
inputFeatures = []
bin_compression_factor = 2
for go_time, end_time in zip(go_cue_nsp_neural_time, trial_end_nsp_neural_time):

    go_ind = np.argmin(np.abs(binned_neural_nsp_timestamp - go_time))
    end_ind = np.argmin(np.abs(binned_neural_nsp_timestamp - end_time))
    
    thresholdCrossings = binned_neural_threshold_crossings[go_ind:end_ind].astype(np.float32)
    spikePower = binned_neural_spike_band_power[go_ind:end_ind]
    spikePower[spikePower>50000]=50000

    # concatenate and append
    newInputFeatures = np.concatenate([thresholdCrossings, spikePower], axis=1)
    inputFeatures.append(newInputFeatures)

# get blockMean and blockStd
blockMean = np.mean(np.concatenate(inputFeatures, 0), axis=0, keepdims=True).astype(np.float32)
blockStd = np.std(np.concatenate(inputFeatures, 0), axis=0, keepdims=True).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------

if include_audio==1:
    # get analog audio signal and timestamps
    stream = r.xrange(analog_stream_name)

    microphone_data = []
    microphone_nsp_time = []

    for entry_id, entry in stream:
        microphone_data.append(np.frombuffer(entry[b'samples'], np.int16).reshape(-1,2)[:,0])
        microphone_nsp_time.append(np.frombuffer(entry[b'timestamps'], dtype=np.uint64))

    microphone_data = np.array(microphone_data).reshape(-1, 1)
    microphone_nsp_time = np.array(microphone_nsp_time).reshape(-1, 1)

    # trim analog data to begin at the start of the first trial, end at the end of the last trial
    nsp_analog_start_time = trial_start_nsp_analog_time[0]
    nsp_analog_end_time = trial_end_nsp_analog_time[-1]

    microphone_start_ind = np.argmin(np.abs(microphone_nsp_time - nsp_analog_start_time))
    microphone_end_ind = np.argmin(np.abs(microphone_nsp_time - nsp_analog_end_time))

    microphone_data = microphone_data[microphone_start_ind:microphone_end_ind]
    microphone_nsp_time = microphone_nsp_time[microphone_start_ind:microphone_end_ind]

else:
    microphone_data = []
    microphone_nsp_time = []

# ---------------------------------------------------------------------------------------------------------------

# initialzie RNN decoder output stream
stream = r.xrange(decoder_output_stream_name)

# if the stream doesn't exist (open loop), make variables blank
if stream==[]:
    decoder_logit_output = np.zeros([1, 41], dtype=np.float32)
    decoder_signal = np.zeros([2], dtype=object)
    decoder_output_redis_clock = np.zeros([1], dtype=np.int64)
    ngram_decoder_partial_output = np.zeros([1], dtype=object)
    ngram_decoder_final_output = np.zeros([1], dtype=object)

# if stream does exist (closed loop), extract decoding info to variables
else:
    decoder_logit_output = np.zeros([len(stream), 41], dtype=np.float32)
    decoder_signal = np.zeros([len(stream), 2], dtype=object)
    decoder_output_redis_clock = np.zeros([len(stream)], dtype=np.int64)
    ngram_decoder_partial_output = np.zeros([len(stream)], dtype=object)
    ngram_decoder_final_output = np.zeros([len(stream)], dtype=object)
    for x in range(len(stream)):
        if b'start' in stream[x][1]:
            decoder_signal[x][0] = stream[x][1][b'start'].decode('utf-8')
        elif b'end' in stream[x][1]:
            decoder_signal[x][1] = stream[x][1][b'end'].decode('utf-8')
        elif b'logits' in stream[x][1]:
            buf = np.frombuffer(stream[x][1][b'logits'], dtype=np.float32)
            decoder_logit_output[x, :buf.shape[0]] = buf
        elif b'partial_decoded_sentence' in stream[x][1]:
            decoded = stream[x][1][b'partial_decoded_sentence'].decode('utf-8')
            ngram_decoder_partial_output[x] = decoded
        elif b'final_decoded_sentence' in stream[x][1]:
            decoded = stream[x][1][b'final_decoded_sentence'].decode('utf-8')
            ngram_decoder_final_output[x] = decoded
        decoder_output_redis_clock_ = str(stream[x][0].decode()).split('-') #get rid of the hyphenated part of the redis timestammp
        decoder_output_redis_clock[x] = int(decoder_output_redis_clock_[0])

# ---------------------------------------------------------------------------------------------------------------
# Get normalization stats from redis brainToText

stream = r.xrange(norm_stats_stream_name)

norm_channel_means = []
norm_channel_stds = []
norm_redis_times = []

if stream != []:
    for entry_id, entry_dict in stream:
        norm_redis_times.append(int(entry_id.decode().split('-')[0]))
        norm_channel_means.append(np.frombuffer(entry_dict[b'mean'], dtype=np.float32))
        norm_channel_stds.append(np.frombuffer(entry_dict[b'std'], dtype=np.float32))


# ---------------------------------------------------------------------------------------------------------------

# Create a dictionary with neural features, task info, and decoder output
keyDict = {
    'participant': participant,
    'session_name': session_name,
    'session_description': session_description,
    'block_number': block_num,
    'block_description': block_decription,
    'block_start_time': start_time,
    'graph_name': graph_name,
    'binned_neural_threshold_crossings': binned_neural_threshold_crossings,
    'binned_neural_spike_band_power': binned_neural_spike_band_power,
    'binned_neural_nsp_timestamp': binned_neural_nsp_timestamp,
    'binned_neural_redis_clock': binned_neural_redis_clock,
    'decoder_logit_output': decoder_logit_output,
    'decoder_signal': decoder_signal,
    'decoder_output_redis_clock': decoder_output_redis_clock,
    'ngram_decoder_partial_output': ngram_decoder_partial_output,
    'ngram_decoder_final_output': ngram_decoder_final_output,
    'norm_channel_means': norm_channel_means,
    'norm_channel_stds': norm_channel_stds,
    'norm_redis_times': norm_redis_times,
    'cue': cue,
    'trial_paused_by_CNRA': trial_paused_by_CNRA,
    'trial_timed_out': trial_timed_out,
    'delay_duration_ms': delay_duration,
    'inter_trial_duration_ms': inter_trial_duration,
    'trial_start_redis_time': trial_start_redis_time,
    'trial_start_nsp_neural_time': trial_start_nsp_neural_time,
    'trial_start_nsp_analog_time': trial_start_nsp_analog_time,
    'go_cue_redis_time': go_cue_redis_time,
    'go_cue_nsp_neural_time': go_cue_nsp_neural_time,
    'go_cue_nsp_analog_time': go_cue_nsp_analog_time,
    'trial_end_redis_time': trial_end_redis_time,
    'trial_end_nsp_neural_time': trial_end_nsp_neural_time,
    'trial_end_nsp_analog_time': trial_end_nsp_analog_time,
    'microphone_data': microphone_data,
    'microphone_nsp_time': microphone_nsp_time,
    'trial_accuracy_confirmation': trial_accuracy_confirmation,
}

# ---------------------------------------------------------------------------------------------------------------

# if root save not specified, get the supergraph's save location for rdb files to use as reference for our save location
if save_location is None:
    save_filename = r.config_get('dbfilename')['dbfilename']
    save_filename = os.path.splitext(save_filename)[0] + '.mat'
    save_filepath = r.config_get('dir')['dir']
    save_filepath = os.path.dirname(save_filepath)
    save_filepath = os.path.join(save_filepath, 'GRU_Training_Files')
    if save_filepath[0:4] == '/mnt':
        save_filepath = save_filepath[4:]

else: # if root save location is specified, use that instead
    save_filepath = Path(save_location, participant, session_name, 'GRU_Training_Files').resolve()
    timeTag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_filename = f'{timeTag}_({block_num}).mat'
    
# create save path if it doesn't exist
os.makedirs(save_filepath, exist_ok=True)

# save dictionary as a .mat file to the desired path, with the desired block number
redismat_path = os.path.join(save_filepath, 'RedisMat')
os.makedirs(redismat_path, exist_ok=True)
fullPath = str(Path(redismat_path, save_filename).resolve())

logging.info('Attempting to save RedisMat data to: ' + fullPath)
scipy.io.savemat(fullPath, keyDict)
logging.info('Succesfully formatted and saved data to: ' + fullPath)

# save blockMean and blockStd
tfdata_path = os.path.join(save_filepath, 'tfdata')
os.makedirs(tfdata_path, exist_ok=True)
save_image_path = os.path.join(save_filepath, 'post_graph_images')
os.makedirs(save_image_path, exist_ok=True)
savePath_blockMean = str(Path(tfdata_path, f'updated_means_block({block_num})').resolve())
os.makedirs(savePath_blockMean, exist_ok=True)

logging.info('Attempting to save rdbToMat_blockMeans to: ' + savePath_blockMean)
np.save(os.path.join(savePath_blockMean, 'rdbToMat_blockMean'), np.squeeze(blockMean, 0))
np.save(os.path.join(savePath_blockMean, 'rdbToMat_blockStd'), np.squeeze(blockStd)+1e-8)
logging.info('Successfully saved rdbToMat_blockMeans to: ' + savePath_blockMean)


# normalize neural features before plotting
binned_neural_threshold_crossings_norm = (binned_neural_threshold_crossings - np.mean(binned_neural_threshold_crossings, axis=0)) \
    / (np.std(binned_neural_threshold_crossings, axis=0) + 1e-8)
binned_neural_spike_band_power_norm = (binned_neural_spike_band_power - np.mean(binned_neural_spike_band_power, axis=0)) \
    / (np.std(binned_neural_spike_band_power, axis=0) + 1e-8)
plt.ioff()
# threshold crossings
plt.figure(1, figsize=(36,16))
plt.subplot(3,1,1)
plt.imshow(binned_neural_threshold_crossings_norm.T, aspect='auto', clim=(-2,2))
for i in range(len(trial_start_nsp_neural_time)):
    trial_start_ind = np.argmin(np.abs(binned_neural_nsp_timestamp - trial_start_nsp_neural_time[i]))
    go_cue_ind      = np.argmin(np.abs(binned_neural_nsp_timestamp - go_cue_nsp_neural_time[i]))
    trial_end_ind   = np.argmin(np.abs(binned_neural_nsp_timestamp - trial_end_nsp_neural_time[i]))
    plt.plot([trial_start_ind, trial_start_ind], [0, n_channels], 'b')
    plt.plot([go_cue_ind, go_cue_ind], [0, n_channels], 'g')
    plt.plot([trial_end_ind, trial_end_ind], [0, n_channels], 'r')
plt.xlabel('Bin #')
plt.ylabel('Channel #')
plt.title('Threshold Crossings (z-scored)')

# spike power
plt.subplot(3,1,2)
plt.imshow(binned_neural_spike_band_power_norm.T, aspect='auto', clim=(-2,2))
for i in range(len(trial_start_nsp_neural_time)):
    trial_start_ind = np.argmin(np.abs(binned_neural_nsp_timestamp - trial_start_nsp_neural_time[i]))
    go_cue_ind      = np.argmin(np.abs(binned_neural_nsp_timestamp - go_cue_nsp_neural_time[i]))
    trial_end_ind   = np.argmin(np.abs(binned_neural_nsp_timestamp - trial_end_nsp_neural_time[i]))
    plt.plot([trial_start_ind, trial_start_ind], [0, n_channels], 'b')
    plt.plot([go_cue_ind, go_cue_ind], [0, n_channels], 'g')
    plt.plot([trial_end_ind, trial_end_ind], [0, n_channels], 'r')
plt.xlabel('Bin #')
plt.ylabel('Channel #')
plt.title('Spike band power (z-scored)')

# microphone signal
if len(microphone_data) > 0:
    plt.subplot(3,1,3)
    plt.plot(microphone_data)
    ylimits = plt.ylim()
    for i in range(len(trial_start_nsp_analog_time)):
        trial_start_ind = np.argmin(np.abs(microphone_nsp_time - trial_start_nsp_analog_time[i]))
        go_cue_ind      = np.argmin(np.abs(microphone_nsp_time - go_cue_nsp_analog_time[i]))
        trial_end_ind   = np.argmin(np.abs(microphone_nsp_time - trial_end_nsp_analog_time[i]))
        plt.plot([trial_start_ind, trial_start_ind], ylimits, 'b')
        plt.plot([go_cue_ind, go_cue_ind], ylimits, 'g')
        plt.plot([trial_end_ind, trial_end_ind], ylimits, 'r')
    plt.xlim(0, len(microphone_data))
    plt.xlabel('Sample #')
    plt.ylabel('Amplitude')
    plt.title('Microphone signal')

plt.suptitle(f'RDB_TO_MAT.PY - {session_name} - BLOCK #{block_num}')

# save the plot
# plt.tight_layout()
plot_savetag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_(' + str(block_num) + ').png'
plot_fullPath = str(Path(save_image_path, plot_savetag).resolve())
plt.savefig(plot_fullPath, bbox_inches='tight')
logging.info('Saved plot to: ' + plot_fullPath)

