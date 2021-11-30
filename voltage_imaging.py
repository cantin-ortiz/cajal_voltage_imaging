import numpy as np
import pandas as pd
import requests as rs

file_to_ignore = ['20211129_M_B_1']
# file_to_ignore = []

def loadMeta(url_spreadsheet = 'https://docs.google.com/spreadsheets/d/1L1ji33YkJ6UJPZMAR22i2lVJndwtUUE-JSUSqi0mnN0/export?format=tsv&id=1L1ji33YkJ6UJPZMAR22i2lVJndwtUUE-JSUSqi0mnN0&gid=0'):
    
    #Redownloading the latest version of the spreadsheet
    path_table = 'meta.tsv'
    res = rs.get(url=url_spreadsheet)
    open(path_table, 'wb').write(res.content)
    #Loading data frame
    meta = pd.read_csv(path_table, header = 0,  sep = '\t')
    #Removing NaN
    meta = meta[~pd.isna(meta['File'])]
    #Removing files to ignore
    meta = meta[~meta['File'].isin(file_to_ignore)]
    return meta 


def getRecordingData(dict_data, r, return_raster = False):
    
    f = dict_data[r['File']]
    list_valid_keys = np.array(list(f.keys()))
    list_valid_keys = list_valid_keys[[s.startswith('I') for s in list_valid_keys]]
    if not return_raster:
        list_valid_keys = list_valid_keys[[int(s.split('_')[1]) == 1 for s in list_valid_keys]]
        selected_key = list_valid_keys[[int(s.split('_')[0][2:]) == r['Measure_ID'] for s in list_valid_keys]][0]
    else:
        selected_key = list_valid_keys[[int(s.split('_')[0][2:]) == r['Measure_ID'] for s in list_valid_keys]]
        selected_key = selected_key[-1]
    return dict_data[r['File']][selected_key]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


#Return the data from a kymo file (.mat)
def getDataKymo(cur_file):
    
    #Discarding meta keys, starting with ___
    all_keys = np.array(list(cur_file.keys()))
    valid_keys = all_keys[[not k.startswith('_') for k in all_keys]]
    
    #Making sure the data has the expected 'size'
    if not len(valid_keys) == 1:
        sys.exit('Issue with the number of valid keys')
    
    data = cur_file[valid_keys[0]]
    return (data)


def movingZscore(signal, half_window_size = 25, std_whole_recording = False):
    
    vect_zscored = np.zeros(len(signal) - 2*half_window_size)
    for i in range(len(vect_zscored)):
        start_point = i
        end_point = i+2*half_window_size+1
        sub_signal = signal[start_point:end_point]
        
        if std_whole_recording:
            vect_zscored[i] = (signal[i+half_window_size] - np.mean(sub_signal)) 
        else:
            vect_zscored[i] = (signal[i+half_window_size] - np.mean(sub_signal)) / np.std(sub_signal)
            
    if std_whole_recording:
        vect_zscored = vect_zscored / np.std(vect_zscored)
        
    return(vect_zscored)


def peakDetection(zscore_signal, threshold, invert_signal = False):
    
    if invert_signal:
        signal = -1*zscore_signal.copy() 
    else:
        signal = zscore_signal.copy()
    peaks = np.where(signal > threshold)[0]
    keep_val = np.ones(len(peaks), dtype = 'bool')
    keep_val[np.where(np.diff(peaks) == 1)[0]+1] = False
    valid_peaks = peaks[keep_val]
    return valid_peaks