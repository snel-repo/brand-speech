# This script takes audio in a .mat file and converts it to a .wav file

import argparse
import glob
import logging
import numpy as np
import os
import scipy.io as sio
import sys

loglevel = 'INFO'
numeric_level = getattr(logging, 'INFO', None)
logging.basicConfig(format='[mat_to_wav] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)

ap = argparse.ArgumentParser(description='Convert .mat file to .wav file')
ap.add_argument('-d', '--dir', required=False, help='Directory of .mat file(s)')
ap.add_argument('-f', '--file', required=False, help='Name of .mat file')
ap.add_argument('-s', '--sample_rate_variable', required=False, default='fs', help='Name of sample rate variable in .mat file')
ap.add_argument('-a', '--audio_variable', required=False, default='y', help='Name of audio variable in .mat file')
ap.add_argument('-t', '--datatype', required=False, default='int16', help='Datatype of output audio')
ap.add_argument('-m', '--max_volume', required=False, default=100, help='Maximum volume as a percentage of the output dtype range')
args = ap.parse_args()

dir = args.dir
file = args.file
sample_rate_var = args.sample_rate_variable
audio_var = args.audio_variable
dtype = args.datatype
max_volume = args.max_volume

if not ((dir is None) ^ (file is None)):
    logging.error('Please provide either a directory or a file name')
    sys.exit(1)

if dir is not None:
    # read in .mat files
    mat_files = glob.glob(os.path.join(dir, '*.mat'))

elif dir is not None:
    mat_files = [file]

dtype = np.dtype(dtype)

def convert_mat_to_wav(mat_file):
    logging.info(f'Converting {mat_file} to .wav')
    mat = sio.loadmat(mat_file)
    audio = mat[audio_var]
    fs = mat[sample_rate_var]
    if isinstance(fs, np.ndarray):
        fs = fs.item()
    wav_file = mat_file.replace('.mat', '.wav')
    audio = (audio / np.abs(audio).max()) * np.iinfo(dtype).max * max_volume/100.
    sio.wavfile.write(wav_file, fs, audio.astype(dtype))

for mat_file in mat_files:
    convert_mat_to_wav(mat_file)

logging.info('Done')