from collections import OrderedDict, defaultdict
from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from parse import get_parser
import pickle
from sklearn.manifold import TSNE
from math import sin, cos, sqrt, atan2, radians


class Model:
    def __init__(self, batchsize, nuser, nitem, duser, ditem, dcateg=0, nlayer=0, nhidden=50):
        demb = np.sqrt(duser + ditem + max(len(userattrs), len(itemattrs)) * dcateg)
        userbatch, itembatch = [], []
        
        with tf.variable_scope('useritem'):
            self.userembs = tf.get_variable(name='userembs', shape=(nuser, duser), dtype=tf.float32, trainable=True,
                                            initializer=tf.random_normal_initializer(stddev=1 / np.sqrt(demb)))
            self.itemembs = tf.get_variable(name='itemembs', shape=(nitem, ditem), dtype=tf.float32, trainable=True,
                                            initializer=tf.random_normal_initializer(stddev=1 / np.sqrt(demb)))
            self.userids = tf.placeholder(tf.int32, shape=(batchsize,), name='userids')
            self.itemids = tf.placeholder(tf.int32, shape=(batchsize,), name='itemids')
            if duser > 0:
                userbatch.append(tf.gather(self.userembs, self.userids, name='userbatch'))
                itembatch.append(tf.gather(self.itemembs, self.itemids, name='itembatch'))
    
        with tf.variable_scope('categorical'):
            self.categembs = {}
            categrefs = {}
            self.categs = {}
            self.usercategrefs = {}
            self.itemcategrefs = {}
            self.usercategids = {}
            self.itemcategids = {}
            usercategbatch = []
            itemcategbatch = []
            allattrs = set(userattrs).union(set(itemattrs))
            print(f'attributes that we will use as covariates {allattrs}')
            for attr in allattrs:
                normattr = normalize_name(attr)
                with tf.variable_scope(normattr):
                    categs = set(users.get(attr, [])).union(set(items.get(attr, [])))
                    categs = list(set(normalize_name(categ) for categ in categs))
                    self.categs[normattr] = categs
                    print(f'embedding all categories from attribute {attr}, {len(categs)} categories found')
                    self.categembs[normattr] = tf.get_variable(name=f'categembs', shape=(len(categs), dcateg), dtype=tf.float32, trainable=True,
                                                          initializer=tf.random_normal_initializer(stddev=1 / np.sqrt(demb)))
                    self.usercategids[normattr] = tf.placeholder(tf.int32, shape=(batchsize,), name=f'usercategids')
                    self.itemcategids[normattr] = tf.placeholder(tf.int32, shape=(batchsize,), name=f'itemcategids')
                    usercategbatch.append(tf.gather(self.categembs[normattr], self.usercategids[normattr], name=f'usercategbatch'))
                    itemcategbatch.append(tf.gather(self.categembs[normattr], self.itemcategids[normattr], name=f'itemcategbatch'))
                categrefs[normattr] = {categ: i for i, categ in enumerate(categs)}
                self.usercategrefs[normattr] = {userid: categrefs[normattr][normalize_name(categ)] for userid, categ in enumerate(users[attr] if attr in users else [])}
                self.itemcategrefs[normattr] = {itemid: categrefs[normattr][normalize_name(categ)] for itemid, categ in enumerate(items[attr] if attr in items else [])}
            if dcateg > 0:
                userbatch.append(tf.concat(usercategbatch, axis=1, name='userconcat'))
                itembatch.append(tf.concat(itemcategbatch, axis=1, name='itemconcat'))
                
        userbatch = tf.concat(userbatch, axis=1, name='userconcat')
        itembatch = tf.concat(itembatch, axis=1, name='itemconcat')
        
        with tf.variable_scope('forward'):
            def forward(x, scope):
                with tf.variable_scope(scope):
                    for layer in range(nlayer):
                        x = tf.layers.dense(x, nhidden, activation=None if layer == nlayer - 1 else tf.nn.relu, use_bias=True, name=f'fc{layer}')
                    return x
            userbatch = forward(userbatch, 'usernet')
            itembatch = forward(itembatch, 'itemnet')
            self.userlogits = userbatch
            self.itemlogits = itembatch

        with tf.variable_scope('losses'):
            self.predbatch = tf.reduce_sum(userbatch * itembatch, axis=1, name='preddist')
            self.truebatch = tf.placeholder(dtype=tf.float32, shape=(batchsize), name='truedist')
            self.loss = tf.reduce_sum((self.predbatch - self.truebatch) ** 2, name='loss')
            self.l1mean = tf.reduce_mean(tf.abs(self.predbatch - self.truebatch))
        
        self.lrnrate = tf.placeholder(tf.float32, shape=(), name='lrnrate')
        self.trainop = tf.train.AdamOptimizer(learning_rate=self.lrnrate).minimize(self.loss)


    def get_categids(self, userids, useritem='user'):
        if useritem == 'user': categrefs = self.usercategrefs
        else: categrefs = self.itemcategrefs
        categids = defaultdict(list)
        for attr in userattrs:
            normattr = normalize_name(attr)
            for userid in userids:
                categids[normattr].append(categrefs[normattr][userid])
        return categids


    def make_feeddict(self, idsbatch, rtnorm):
        userids, itemids = idsbatch[:, 0], idsbatch[:, 1]
        usercategids = self.get_categids(userids, 'user')
        itemcategids = self.get_categids(itemids, 'item')
        truebatch = np.array([rtnorm[userid, itemid] for userid, itemid in idsbatch])
        feeddict = {self.userids: userids, self.itemids: itemids, self.truebatch: truebatch,
                    **{self.usercategids[key]: val for key, val in usercategids.items()},
                    **{self.itemcategids[key]: val for key, val in itemcategids.items()}}
        return feeddict


    def get_truebatch(self, idsbatch, rtnorm):
        truebatch = np.array([rtnorm[userid, itemid] for userid, itemid in idsbatch])
        return truebatch


def valid(epoch, step):
    losses, preds, trues = [], [], []
    for i in range(0, len(validids) - args.batchsize + 1, args.batchsize):
        idsbatch = validids[i: i + args.batchsize]
        l1, predbatch = sess.run([model.l1mean, model.predbatch], model.make_feeddict(idsbatch, rtnorm))
        losses.append(l1)
        preds.extend(list(predbatch)[:20])
        trues.extend(list(model.get_truebatch(idsbatch, rtnorm))[:20])
    experiment.log_metric('l1V', l1, step=step)
    trues, preds = np.array(trues), np.array(preds)
    trues, preds = trues * std + mean, preds * std + mean
    if epoch in [0, args.nepoch - 1]: plt.plot(trues, preds, '.r' if epoch == 0 else '.b', alpha=.3, markeredgewidth=0, label='untrained' if epoch == 0 else 'trained')
    print(f'valid | epoch {epoch} | loss {np.mean(losses)}')
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, xlim, '-g')
    plt.xlabel('ground truth')
    plt.ylabel('predicted')
    plt.gca().axis('equal')
    plt.title('log response time')
    plt.legend()
    plt.tight_layout()
    experiment.log_figure(step=epoch)


def train(epoch, step):
    for i in range(0, len(trainids) - args.batchsize + 1, args.batchsize):
        feeddict = model.make_feeddict(trainids[i: i + args.batchsize], rtnorm)
        feeddict.update({model.lrnrate: get_lrnrate(step, lrnrate=args.lrnrate)})
        _, l1 = sess.run([model.trainop, model.l1mean], feeddict)
        if not step % 10:
            experiment.log_metric('l1', l1, step=step)
            print(f'train | epoch {epoch} | step {step} | loss {l1}')
        step += 1
    return step


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # data
    userattrs = itemattrs = ['subnet', 'Country', 'AS']
    rt, users, items, nuser, nitem = load_data()
    users, items = extract_feats(users, items)
    trainids, validids = extract_pair_ids(rt, nuser, nitem, splitratio=args.splitratio)
    rtnorm, mean, std = inverse_standardized_log_latency(rt)
    plt.hist(rtnorm.ravel(), 100)
    plt.savefig('debug.png')
    plt.close()

    # model
    model = Model(args.batchsize, nuser, nitem, args.duser, args.ditem, args.dcateg, args.nlayer, args.nhidden)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('./', graph=sess.graph)
    
    # begin training
    experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", project_name='wsdream', workspace='wronnyhuang', display_summary=False)
    plt.figure(figsize=(5, 5))
    step = 0
    for epoch in range(args.nepoch):
        valid(epoch, step)
        step = train(epoch, step)

    
    ## embedding tsne visualizations
    # country
    categembs = sess.run(model.categembs)
    with open('categembs.pkl', 'wb') as f: pickle.dump(categembs, f)
    with open('categembs.pkl', 'rb') as f: categembs = pickle.load(f)
    embs = categembs['country']
    tsnes = TSNE(n_components=2).fit_transform(embs)
    plt.figure(figsize=(8, 8))
    plt.plot(*tsnes.T, '.')
    for i, tsne in enumerate(tsnes):
        plt.text(*tsne, ' ' + model.categs['country'][i], fontsize=8)
    plt.gca().axis('equal')
    plt.tight_layout()
    print(experiment.log_figure(step=epoch))

    # AS
    embs = categembs['as'][:300]
    tsnes = TSNE(n_components=2).fit_transform(embs)
    plt.figure(figsize=(16, 16))
    plt.plot(*tsnes.T, '.')
    for i, tsne in enumerate(tsnes):
        plt.text(*tsne, ' ' + model.categs['as'][i][3:23], fontsize=8)
    plt.gca().axis('equal')
    plt.tight_layout()
    print(experiment.log_figure(step=epoch))

    # subnet
    embs = categembs['subnet']
    tsnes = TSNE(n_components=2).fit_transform(embs)
    plt.figure(figsize=(8, 8))
    plt.plot(*tsnes.T, '.')
    for i, tsne in enumerate(tsnes):
        plt.text(*tsne, ' ' + model.categs['subnet'][i], fontsize=8)
    plt.gca().axis('equal')
    plt.tight_layout()
    print(experiment.log_figure(step=epoch))
    
    ## correlation between latency and distance (hint: none)
    def latlondist(lat1, lon1, lat2, lon2):
        # approximate radius of earth in km
        R = 6373.0 * 1e-3
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    lldists = []
    latencies = []
    for userid, itemid in trainids[:500]:
        lat1, lon1 = users['Latitude'][userid], users['Longitude'][userid]
        lat2, lon2 = items['Latitude'][itemid], items['Longitude'][itemid]
        lldists.append(latlondist(lat1, lon1, lat2, lon2))
        latencies.append(np.log10(rt[userid, itemid]))
    plt.figure(figsize=(5, 5))
    plt.plot(lldists, latencies, '.')
    plt.title('relationship between physical distance and latency')
    plt.xlabel('physical distance (km)')
    plt.ylabel('log response time (s)')
    print(experiment.log_figure())
    print(f'time for light to circle the earth inside silica fiber: {40e3 / 3e8 * 1.5 * 1000} ms')
