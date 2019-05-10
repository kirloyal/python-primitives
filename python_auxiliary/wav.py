import os
import numpy as np
import scipy.io.wavfile

if not os.path.exists('tmp'):
    os.makedirs('tmp')

data = 10 * np.sin(np.linspace(0, 440 * 2 * np.pi, num=48000))
scipy.io.wavfile.write('tmp/tmp.wav', 48000, data)

#%%

rate, data_read = scipy.io.wavfile.read('tmp/tmp.wav')
print(rate, data_read.shape)