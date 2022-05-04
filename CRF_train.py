import pandas as pd
import pickle
import numpy as np
import re
import json
import regex
from ast import literal_eval
from nltk import pos_tag
from nervaluate import Evaluator
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import pickle
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical
from tqdm.keras import TqdmCallback
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from tqdm.notebook import tqdm, trange

from nervaluate import Evaluator
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from utils import *



# ---------- Load train test split -----------
# train_pids, valid_pids, test_pids, unseen_pids = pickle.load(open("train_test_split.p", "rb"))
train_pids, valid_pids, test_pids, test_pids_cat = pickle.load(open("./data/train_test_split_0331.p", "rb"))
train_idxs, valid_idxs, test_idxs, unseen_idxs = [], [], [], []

unseen_pids = test_pids_cat["unseen"] + valid_pids.tolist()
for k in test_pids_cat.keys():
    if k == "unseen":
        continue
    unseen_pids = [i for i in unseen_pids if i not in test_pids_cat[k]]
len(test_pids_cat["unseen"]), len(unseen_pids)

#----------- load dataset------------
input_file = "./data/ml_datasetname_inputs_flv0.p"
X, y, X_pids = pickle.load(open(input_file, "rb"))

for i in range(len(X_pids)):
    if X_pids[i] in train_pids:
        train_idxs.append(i)
    elif X_pids[i] in valid_pids:
        valid_idxs.append(i)
    elif X_pids[i] in test_pids:
        test_idxs.append(i)
    if X_pids[i] in unseen_pids:
        unseen_idxs.append(i)
        
tot = len(train_idxs) + len(valid_idxs) + len(test_idxs)
print(f"nSamples: train={len(train_idxs):,} ({len(train_idxs)*100/tot:.2f}%), valid={len(valid_idxs):,} ({len(valid_idxs)*100/tot:.2f}%)")
print(f"test={len(test_idxs):,} ({len(test_idxs)*100/tot:.2f}%), unseen = {len(unseen_idxs):,} ({len(unseen_idxs)*100/tot:.2f}%)")



#-------- load processed CRF X file------
# load processed data
X = pickle.load(open("./data/CRF_X.p", "rb"))


# ----------- spliting -------------
X_tr = np.array([X[i] for i in train_idxs])
X_val = np.array([X[i] for i in valid_idxs])
X_te = np.array([X[i] for i in test_idxs])
X_te_seen = np.array([X[i] for i in test_idxs if i not in unseen_idxs])
X_te_unseen = np.array([X[i] for i in test_idxs if i in unseen_idxs])

y_tr = np.array([y[i] for i in train_idxs])
y_val = np.array([y[i] for i in valid_idxs])
y_te = np.array([y[i] for i in test_idxs])
y_te_unseen = np.array([y[i] for i in test_idxs if i in unseen_idxs])
y_te_seen = np.array([y[i] for i in test_idxs if i not in unseen_idxs])



#----- training processing------

from sklearn_crfsuite import CRF
crf = CRF(algorithm='lbfgs',
          c1=10,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

size_list = list(range(10000,51000,5000))

pbar = tqdm(total=len(size_list))
for i in size_list:
    pbar.update(1)
    #train with mini batch
    X_tr_batch = X_tr[0:i]
    y_tr_batch = y_tr[0:i]
    crf.fit(X_tr_batch, y_tr_batch)
    # Test for unseen
    test_pred = crf.predict(X_te_unseen)
    preds = [[j if j is not None else 'O' for j in i] for i in test_pred]
    test_labels = [np.array(i).astype('<U1').tolist() for i in y_te_unseen]
    
    report = classification_report(test_labels, preds)
    training_size = "Training size:"+(str(len(X_tr_batch)))
    with open('crf_train.txt', "a") as file:
        file.write('\n')
        file.write(training_size)
        file.write('\n')
        file.write(report)
        file.write('\n')
    file.close()
    
    


