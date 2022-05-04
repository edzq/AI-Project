import os
import math
import random
import csv
import sys
import pickle
sys.path.append(os.getcwd() + "/bert-sklearn/")
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

# ---------- Load inputs-----------
#input_file = "../data/ml_datasetname_inputs_flv0.p"
X, y, X_pids = pickle.load(open("./data/ml_datasetname_inputs_flv0.p", "rb"))

#-----Jo's Split------
train_pids, valid_pids, test_pids, unseen_pids = pickle.load(open("./data/train_test_split.p", "rb"))
train_idxs, valid_idxs, test_idxs = [], [], []

for i in range(len(X_pids)):
    if X_pids[i] in train_pids:
        train_idxs.append(i)
    elif X_pids[i] in valid_pids:
        valid_idxs.append(i)
    elif X_pids[i] in test_pids:
        test_idxs.append(i)
        
X_tr = np.array([X[i] for i in train_idxs])
X_val = np.array([X[i] for i in valid_idxs])
X_te = np.array([X[i] for i in test_idxs])

y_tr = np.array([y[i] for i in train_idxs])
y_val = np.array([y[i] for i in valid_idxs])
y_te = np.array([y[i] for i in test_idxs])

print(f"nSamples: train={len(X_tr):,}, valid={len(X_val):,}, test={len(X_te):,}")


#testing save file
#X_tr = X_tr[0:100]
#y_tr = y_tr[0:100]

#X_te = X_te[0:100]
#y_te = y_te[0:100]



#run the Sci-BERT
#%%time
label_list = ['B', 'I', 'O']
# define model

# Choose between BERT or SciBERT

model = BertTokenClassifier(bert_model='scibert-scivocab-cased',
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

# finetune model
model.fit(np.array(X_tr), np.array(y_tr))

# # score model
f1_test = model.score(X_te, y_te, 'macro')
print("Test f1: %0.02f"%(f1_test))

# make predictions
y_preds = model.predict(np.array(X_te))

#----save the model---
#savefile = 'scibert_Jo_split.bin'
#model.save(savefile)

#----print test restul
evaluator = Evaluator(y_te, y_preds, tags= [''], loader='list')
results, results_per_tag = evaluator.evaluate()
print(results)

#----save the model---
savefile = 'scibert_Jo_split.bin'
model.save(savefile)
