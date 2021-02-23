import librosa
import numpy as np
import os
import timeit
import concurrent.futures

 
def create_spectrograms(duration, np_size, data_path, save_path): 
    u"""
    creates spectrogram representations of all audio files at given duration
    duration: how many seconds will be taken from each audio clip 
    np_size: size of resulting np array based on duration 
    data_path: file path for directory containing subdirectories of each audio file based on language
    save_path: location to save processed data 
    """
    #initialize np arrays 
    train_x = np.empty((0, 128, np_size), int)
    train_y = np.array([])

    test_x = np.empty((0, 128, np_size), int)
    test_y = np.array([])

    languages = [f for f in os.listdir(data_path) if not f.startswith('.')]

    for language in languages:
        print(f"Starting {language} for {duration} seconds")
        # randomize data 
        data = os.listdir(data_path + '/' + language)
        np.random.shuffle(data)
        n = len(data)
        count = 0
        
        for wav_file in data:
            count += 1
            # create np array of spectrogram from wav file 
            y, sr = librosa.load(data_path + '/' + language + '/' + wav_file, duration=duration)
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            # 70% training data 
            if count <= .7*n: 
                train_x = np.append(train_x, [spec], axis=0)
                train_y = np.append(train_y, language)
            # 30% testing data
            else: 
                test_x = np.append(test_x, [spec], axis=0)
                test_y = np.append(test_y, language)
    # save np arrays 
    np.save(save_path + '/{}seconds/train_x_{}'.format(duration,duration), train_x)
    np.save(save_path + '/{}seconds/train_y_{}'.format(duration,duration), train_y)    
    np.save(save_path + '/{}seconds/test_x_{}'.format(duration,duration), test_x)
    np.save(save_path + '/{}seconds/test_y_{}'.format(duration,duration), test_y)

    return f'Done processing {duration} seconds: \nx_train: {train_x.shape} y_train: {train_y.shape} \nx_test {test_x.shape} y_test{test_y.shape}'


# def process_all(duration, np_size, data_path, save_path): 
#     u"""
#     concurrently creates spectrograms for every audio file 
#     duration: how many seconds will be taken from each audio clip 
#     np_size: size of resulting np array based on duration 
#     data_path: file path for wav data 
#     save_path: location to save processed data  
#     """

#     folders = os.listdir(data_path)[1:]

#     # concurrently create spectrograms of different languages at one length 
#     with concurrent.futures.ThreadPoolExecutor() as executor: 
#         results = [executor.submit(create_spectrograms, duration, np_size, data_path, save_path) for folder in folders]
#         for f in concurrent.futures.as_completed(results): 
#             print(f.result())
        
#     return 'Done processing {} second duration'.format(duration)
    

if __name__ == "__main__":         

    data_path = '/Users/seenahuang/Desktop/AudioLanguageClassifier/wav'
    save_path = '/Users/seenahuang/Desktop/AudioLanguageClassifier'

    # different lengths to evaluate wav files
    durations = [3,5,7]
    # size of resulting np array at respective durations
    np_sizes = [130,216,302]

    start = timeit.default_timer()

    # concurrently process data at different time lengths
    with concurrent.futures.ThreadPoolExecutor() as executor: 
        results = [executor.submit(create_spectrograms,durations[i], np_sizes[i],data_path, save_path) for i in range(len(durations))]
        for f in concurrent.futures.as_completed(results): 
            print(f.result())

    end = timeit.default_timer()
    print("Completed in {} minutes".format((end-start)/60))


 




