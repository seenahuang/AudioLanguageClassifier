import librosa
import numpy as np
import os
import timeit
import concurrent.futures


# process audio files to turn them into np array spectrogram representations 
# duration: how many seconds will be taken from each audio clip 
# np_size: size of resulting np array based on duration 
# data_path: file path for wav data 
# save_path: location to save processed data  
def create_data(duration, np_size, data_path, save_path): 

    #initialize np arrays 
    train_x = np.empty((0, 128, np_size), int)
    train_y = np.array([])

    test_x = np.empty((0, 128, np_size), int)
    test_y = np.array([])

    for folder in os.listdir(data_path)[1:]:
        count = 0
        print('{} {} seconds'.format(folder, duration))

        if folder == '.DS_Store': 
            continue

        # randomize data 
        data = os.listdir(data_path + '/' + folder)
        np.random.shuffle(data)
        n = len(data)

        for wav_file in data:
            count += 1
            # create np array of spectrogram from wav file 
            y, sr = librosa.load(data_path + '/' + folder + '/' + wav_file, duration=duration)
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            # 70% training data 
            if count <= .7*n: 
                train_x = np.append(train_x, [spec], axis=0)
                train_y = np.append(train_y, folder)
            # 30% testing data
            else: 
                test_x = np.append(test_x, [spec], axis=0)
                test_y = np.append(test_y, folder)
    # save np arrays 
    np.save(save_path + '/{}seconds/train_x_{}'.format(duration,duration), train_x)
    np.save(save_path + '/{}seconds/train_y_{}'.format(duration,duration), train_y)    
    np.save(save_path + '/{}seconds/test_x_{}'.format(duration,duration), test_x)
    np.save(save_path + '/{}seconds/test_y_{}'.format(duration,duration), test_y)
    return 'Done processing {} second duration'.format(duration)
    

def main():         

    data_path = '/Users/seenahuang/Desktop/AudioLanguageClassifier/wav'
    save_path = '/Users/seenahuang/Desktop/AudioLanguageClassifier'

    # different lengths to evaluate wav files
    durations = [3,5,7]
    # size of resulting np array at respective durations
    np_sizes = [130,216,302]

    start = timeit.default_timer()

    # concurrently process data at different time lengths
    with concurrent.futures.ThreadPoolExecutor() as executor: 
        results = [executor.submit(create_data,durations[i], np_sizes[i],data_path, save_path) for i in range(len(durations))]
        for f in concurrent.futures.as_completed(results): 
            print(f.result())

    end = timeit.default_timer()
    print("Completed in {} minutes".format((end-start)/60))


if __name__ == "__main__": 
    main()



