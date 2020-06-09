
from os.path import isdir
from os import mkdir
import pickle
def pickle_dump(file_path, obj):
    dest_folder = file_path.rsplit('/',1)[0]
    
    if not isdir(dest_folder): 
        mkdir(dest_folder) # Create folder if it does not exist

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(file_path):
    dest_folder = file_path.rsplit('/',1)[0]

    if not isdir(dest_folder): 
        print(f'{dest_folder} not found')
        return

    with open(file_path, 'rb') as f:
        return pickle.load(f)

def sort_list_dict(_list_dict, by=0, descending=False):
    if by == 'key': by = 0
    elif by == 'value': by = 1
    elif not by in [0, 1, 'key', 'value']: raise ValueError('Invalid')
    return sorted(_list_dict, key=lambda item: item[by], reverse=descending)

def sort_dict(_dict, by=0, descending=False, limit=None):
    '''
        by : {0/'key', 1/'value'}, default 0
    '''
    if by == 'key': by = 0
    elif by == 'value': by = 1
    elif not by in [0, 1, 'key', 'value']: raise ValueError('Invalid')
    
    _sorted = sorted(
        _dict.items(), 
        key=lambda item: item[by], 
        reverse=descending
    )
    
    if limit:
        _sorted = _sorted[:limit]
    
    return {
        k:v 
        for k, v in _sorted
    }

def form_dict(data_df, new_key, new_value):
    inverse_dict = {}
    for keys, value in data_df[[new_key, new_value]].values:
        for key in keys:
            if key not in inverse_dict:
                inverse_dict[key] = [value]
                continue
            inverse_dict[key].append(value)
    return inverse_dict

def chunks(max_length, interval, overlap):
    for idx in list(range(0, max_length, interval))+[max_length]:
        if idx == 0:
            yield idx, idx+int(overlap)
        else:
            yield idx-int(overlap), idx+int(overlap)
