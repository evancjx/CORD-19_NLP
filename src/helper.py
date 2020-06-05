
def sort_list_dict(_list_dict, by, descending=False):
    return sorted(_list_dict, key=lambda item: item[by], reverse=descending)

def sort_dict(_dict, by=0, descending=False):
    '''
        by : {0/'key', 1/'value'}, default 0
    '''
    if by == 'key': by = 0
    elif by == 'value': by = 1
    elif not by in [0, 1, 'key', 'value']: raise ValueError('Invalid')
    
    return {
        k:v 
        for k, v in sorted(
            _dict.items(), 
            key=lambda item: item[by], 
            reverse=descending
        )
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
