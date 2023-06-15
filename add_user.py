import pyaudio
import wave
import os
import pickle
import time
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output
import matplotlib.pyplot as plt

from main_functions import *

def add_user():
    
    name = input("Enter Name:")
    #Voice authentication
    FORMAT = pyaudio.paInt16  # 16-bit binary string
    CHANNELS = 2  # Channel refers to the number of audio streams to use
    RATE = 44100  # the number of samples collected per second
    CHUNK = 1024  # the number of frames in the buffer
    RECORD_SECONDS = 3 
    
    source = "./voice_database/" + name
    
   
    os.mkdir(source)
    for i in range(3):
        audio = pyaudio.PyAudio()

        if i == 0:
            j = 3
            while j>=0: # means the record has ended after 3 times
                time.sleep(1.0)
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Speak your name in {} seconds".format(j))
                j-=1

        elif i ==1:  # means there is a folder
            time.sleep(2.0)
            print("Speak your name one more time")
            time.sleep(0.8)
        
        else:  # means enter your voice print one more time
            time.sleep(2.0)
            print("Speak your name one last time")
            time.sleep(0.8)

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

        print("recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK) # means to read the sample rates and data from each chunck of waves, store it in :data
            frames.append(data)

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # saving wav file of speaker
        waveFile = wave.open(source + '/' + str((i+1)) + '.wav', 'wb') # wave : analyze the wave file
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        print("Done")

    dest =  "./gmm_models/"
    count = 1

    for path in os.listdir(source):
        path = os.path.join(source, path)
        features = np.array([]) # Create an array, Specify the memory layout of the array:either 64-bit signed integers or double precision floating point numbers.

        # reading audio files of speaker
        (sr, audio) = read(path)
        
        # extract 40 dimensional MFCC & delta MFCC features
        vector = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector)) # Stack arrays in sequence vertically (row order)
            
            
        # when features of 3 files of speaker are concatenated, then do model training
        if count == 3:    
            gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
            gmm.fit(features) # 
            #print(f"training model"+{features})

            # saving the trained gaussian model
            pickle.dump(gmm, open(dest + name + '.gmm', 'wb')) # converts a Python object hierarchy into a byte stream
            print(name + ' added successfully') 
            print("training model")
            print(features)
            print("\n")
            features = np.asarray(()) # used when we want to convert input to an array
            count = 0
        count = count + 1

    plt.plot(vector) # to print the plot vector 
    plt.xlabel('Time')
    plt.ylabel('MFCC')
    plt.show()

    #print(f"training model"+{features})
if __name__ == '__main__':
    add_user()
