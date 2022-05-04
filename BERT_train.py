import os
import math
import random
import csv
import sys
import pickle
sys.path.append(os.getcwd() + "/bert-sklearn/")
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm.notebook import tqdm, trange

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
import statistics as stats

from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model
from nervaluate import Evaluator
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from tqdm import trange


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


# ---------- Load inputs-----------
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


# ----------- spliting -------------
X_tr = [X[i] for i in train_idxs]
X_val = [X[i] for i in valid_idxs]
X_te = [X[i] for i in test_idxs]
X_te_seen = [X[i] for i in test_idxs if i not in unseen_idxs]
X_te_unseen = [X[i] for i in test_idxs if i in unseen_idxs]

y_tr = [y[i] for i in train_idxs]
y_val = [y[i] for i in valid_idxs]
y_te = [y[i] for i in test_idxs]
y_te_unseen = [y[i] for i in test_idxs if i in unseen_idxs]
y_te_seen = [y[i] for i in test_idxs if i not in unseen_idxs]

#run the Sci-BERT or BERT
#%%time
label_list = ['B', 'I', 'O']
# define model

# Choose between BERT or SciBERT

model = BertTokenClassifier(bert_model='bert-base-cased',
# model = BertTokenClassifier(bert_model='scibert-scivocab-cased',
                            max_seq_length=178,
                            epochs=3,
                            gradient_accumulation_steps=4,
                            learning_rate=5e-5,
                            train_batch_size=16,
                            eval_batch_size=16,
                            validation_fraction=0., 
                            label_list=label_list,                         
                            ignore_label=['O'])


print(model)

size_list = list(range(10000,51000,5000))

pbar = tqdm(total=len(size_list))
for i in size_list:
    pbar.update(1)
    #train with mini batch
    X_tr_batch = X_tr[0:i]
    y_tr_batch = y_tr[0:i]
    model.fit(np.array(X_tr_batch), np.array(y_tr_batch))
    
    # Test for unseen
    test_pred = model.predict(X_te_unseen)
    preds = [[j if j is not None else 'O' for j in i] for i in test_pred]
    test_labels = [np.array(i).astype('<U1').tolist() for i in y_te_unseen]
    report =classification_report(test_labels, preds)
    training_size = "Training size:"+(str(len(X_tr_batch)))
    with open('BERT_train.txt', "a") as file:
        file.write('\n')
        file.write(training_size)
        file.write('\n')
        file.write(report)
        file.write('\n')
    file.close()
    
    
# ------- test Sci-BERT---------- 
#model = BertTokenClassifier(bert_model='bert-base-cased',
model = BertTokenClassifier(bert_model='scibert-scivocab-cased',
                            max_seq_length=178,
                            epochs=3,
                            gradient_accumulation_steps=4,
                            learning_rate=5e-5,
                            train_batch_size=16,
                            eval_batch_size=16,
                            validation_fraction=0., 
                            label_list=label_list,                         
                            ignore_label=['O'])


print(model)


size_list = list(range(10000,51000,5000))

pbar = tqdm(total=len(size_list))
for i in size_list:
    pbar.update(1)
    #train with mini batch
    X_tr_batch = X_tr[0:i]
    y_tr_batch = y_tr[0:i]
    model.fit(np.array(X_tr_batch), np.array(y_tr_batch))
    
    # Test for unseen
    test_pred = model.predict(X_te_unseen)
    preds = [[j if j is not None else 'O' for j in i] for i in test_pred]
    test_labels = [np.array(i).astype('<U1').tolist() for i in y_te_unseen]
    report =classification_report(test_labels, preds)
    training_size = "Training size:"+(str(len(X_tr_batch)))
    with open('BERT_train.txt', "a") as file:
        file.write('\n')
        file.write(training_size)
        file.write('\n')
        file.write(report)
        file.write('\n')
    file.close()
