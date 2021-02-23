import librosa
import numpy as np
import os
import time



start = time.time()
data_path = '/Users/seenahuang/Desktop/AudioLanguageClassifier/wav'



# different lengths to evaluate wav files
durations = [3,5,7]
# size of resulting np array at respective durations
np_sizes = [130,216,302]

for i,d in enumerate(durations): 
    #initialize np arrays 
    train_x = np.empty((0, 128, np_sizes[i]), int)
    train_y = np.array([])

    test_x = np.empty((0, 128, np_sizes[i]), int)
    test_y = np.array([])

    print('Preprocessing for ' + str(d) + ' seconds')

    for folder in os.listdir(data_path)[1:]:
        count = 0
        print(folder)

        if folder == '.DS_Store': 
            continue

        # randomize data 
        data = os.listdir(data_path + '/' + folder)
        np.random.shuffle(data)
        n = len(data)

        for wav_file in data:
            count += 1
            # create np array of spectrogram from wav file 
            y, sr = librosa.load(data_path + '/' + folder + '/' + wav_file, duration=d)
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            # 70% training data 
            if count <= .7*n: 
                train_x = np.append(train_x, [spec], axis=0)
                train_y = np.append(train_y, folder)
            # 30% testing data
            else: 
                test_x = np.append(test_x, [spec], axis=0)
                test_y = np.append(test_y, folder)
            # show progress 
            if count % 100 == 0:
                print(str(count) + '/' + str(n))
    # save np arrays 
    save_dir = '/Users/seenahuang/Desktop/AudioLanguageClassifier/'
    np.save(save_dir + '/' + str(d) + 'seconds/train_x'+str(i), train_x)
    np.save(save_dir + '/' + str(d) + 'seconds/train_y'+str(i), train_y)    
    np.save(save_dir + '/' + str(d) + 'seconds/test_x'+str(i), test_x)
    np.save(save_dir + '/' + str(d) + 'seconds/test_y'+str(i), test_y) 

    print('train shape: ' + str(train_x.shape))  
    print('train label shape: ' + str(train_y.shape)) 
    print('test shape: ' + str(test_x.shape)) 
    print('test label shape: ' + str(test_x.shape)) 

print(f"Time elapsed: {time.time()-start}")




