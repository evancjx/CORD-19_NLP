
def reconstructText(tokens, start=0, stop=-1):
    tokens = tokens[start: stop]
    if '[SEP]' in tokens:
        sepind = tokens.index('[SEP]')
        tokens = tokens[sepind+1:]
    txt = ' '.join(tokens)
    txt = txt.replace(' ##', '')
    txt = txt.replace('##', '')
    txt = txt.strip()
    txt = " ".join(txt.split())
    txt = txt.replace(' .', '.')
    txt = txt.replace('( ', '(')
    txt = txt.replace(' )', ')')
    txt = txt.replace(' - ', '-')
    txt_list = txt.split(' , ')
    txt = ''
    nTxtL = len(txt_list)
    if nTxtL == 1:
        return txt_list[0]
    newList =[]
    for i,t in enumerate(txt_list):
        if i < nTxtL -1:
            if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                newList += [t,',']
            else:
                newList += [t, ', ']
        else:
            newList += [t]
    return ''.join(newList)