from collections import OrderedDict, defaultdict
import numpy as np
import os

## Download the dataset
def load_data():
    # Parse out latency matrix
    rt = []
    f = open('rtMatrix.txt', 'r')
    line = '\n'
    while '\n' in line:
        line = f.readline()
        values = []
        for i, val in enumerate(line.split('\t')):
            try:
                values.append(float(val))
            except:
                pass
        rt.append(values[:-1])
    f.close()
    rt = np.array(rt[:-1])
    print(f'shape of response time user-service matrix {rt.shape}')
    
    # parse user feats
    users = OrderedDict()
    keys = []
    types = [int, str, str, int, str, float, float]
    f = open('userlist.txt', 'r')
    line = '\n'
    l = 0
    for l in range(len(rt) + 2):
        line = f.readline()
        for i, val in enumerate(line.split('\t')):
            val = val.replace('\n', '')
            if l == 0:
                key = val[1:-1]
                users[key] = []
                keys.append(key)
            elif l > 1:
                users[keys[i]].append(types[i](val))
    f.close()
    nuser = len(users["User ID"])
    print(f'no of users {nuser}, no attributes {len(users)}')
    
    # parse item feats
    def nanfloat(a):
        try:
            return float(a)
        except:
            return np.nan
    
    items = OrderedDict()
    keys = []
    types = [int, str, str, str, str, int, str, nanfloat, nanfloat]
    f = open('wslist.txt', 'r', errors='replace')
    line = '\n'
    l = 0
    for l in range(len(rt[0]) + 2):
        line = f.readline()
        for i, val in enumerate(line.split('\t')):
            val = val.replace('\n', '')
            if l == 0:
                key = val[1:-1]
                items[key] = []
                keys.append(key)
            elif l > 1:
                items[keys[i]].append(types[i](val))
    f.close()
    nitem = len(items["Service ID"])
    print(f'no of services {nitem}, no attributes {len(items)}')
    os.system('head -12 userlist.txt')
    os.system('head -12 wslist.txt')
    os.system('head -5 rtMatrix.txt')
    return rt, users, items, nuser, nitem


def extract_feats(users, items):
    # IP address subnets
    items['subnet'] = []
    for ip in items['IP Address']:
        if '.' in ip:
            subnet = int(ip.split('.')[0])
            items['subnet'].append(subnet)
        else:
            items['subnet'].append(np.nan)
    users['subnet'] = []
    for ip in users['IP Address']:
        if '.' in ip:
            subnet = int(ip.split('.')[0])
            users['subnet'].append(subnet)
        else:
            users['subnet'].append(np.nan)
    return users, items


def normalize_name(name):
    if type(name) is str:
        name = name.replace(' ', '_')
        name = ''.join(char for char in name if char.isalpha() or char == '_')
        name = name.lower()
    return str(name)


def extract_pair_ids(rt, nuser, nitem, splitratio=.5):
    usermat, itemmat = np.meshgrid(range(nuser), range(nitem))
    allpairs = list(zip(usermat.ravel(), itemmat.ravel()))
    allpairs = [[userid, itemid] for userid, itemid in allpairs if rt[userid, itemid] > 0]
    allpairs = np.array(allpairs)
    permutation = np.random.permutation(len(allpairs))
    allpairs = allpairs[permutation]
    split = int(np.floor(splitratio * len(permutation)))
    xtrain = allpairs[:split]
    xvalid = allpairs[split:2*split]
    return xtrain, xvalid


def inverse_standardized_log_latency(rt):
    rt = np.log10(rt)
    mean = np.nanmean(rt)
    rt = rt - mean
    std = np.nanstd(rt)
    rt = rt / std
    rt = -rt
    return rt, mean, std


def get_lrnrate(step, lrnrate=.05, period=500):
    if step < period * 1:
        return lrnrate * 1e0
    elif step < period * 2:
        return lrnrate * 1e-1
    elif step < period * 3:
        return lrnrate * 1e-2
    else:
        return lrnrate * 1e-3
