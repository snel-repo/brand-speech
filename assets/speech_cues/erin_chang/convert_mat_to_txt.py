# %%

import os
import scipy.io as sio

fpath = '/home/samnt/Projects/emory-cart/brand-modules/brand-speech/assets/speech_cues/erin_chang/speech_train_chang_sentences.mat'

# Load the .mat file

mat = sio.loadmat(fpath)

# read each sentence and write to a txt file
write_path = os.path.splitext(fpath)[0] + '.txt'
with open(write_path, 'w') as f:
    for i in range(len(mat['sentences'][0])):
        f.write(mat['sentences'][0][i].item() + '\n')
# %%
