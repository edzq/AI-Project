import pickle
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical
from tqdm.keras import TqdmCallback
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant

from tqdm.notebook import tqdm, trange

from nervaluate import Evaluator
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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


input_file = "./data/ml_datasetname_inputs_flv0.p"
X, y, X_pids = pickle.load(open(input_file, "rb"))


for i in trange(len(X_pids)):
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


# ---------- Preprocess inputs-----------
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent in X:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
word_to_ix["ENDPAD"] = len(word_to_ix) # the corresponding padding
words = word_to_ix.keys()
ix_to_word = dict((v, k) for k, v in word_to_ix.items())

tag_to_ix = {
'O': 0,
'B': 1,
'I': 2,
}

X = [[word_to_ix[w] for w in s] for s in X]
y = [[to_categorical(tag_to_ix[w], num_classes=3) for w in s] for s in y]

max_len = len(X[0])
n_words = len(word_to_ix.keys())
n_tags = len(tag_to_ix.keys())

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word_to_ix["ENDPAD"])
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag_to_ix["O"])
        
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


print(f"nSamples: train={len(X_tr):,}, valid={len(X_val):,}, test={len(X_te):,}")

# ---------- pretrained embedding ----------
embedding_dim = 300 # v1 without pretrain, it was 50

# ---------- pretrained embedding ----------
#https://keras.io/examples/nlp/pretrained_word_embeddings/
PRETRAINED = "glove_300" # "", "GLOVE_V1", "word2vec", "glove_300"
if PRETRAINED != "":
    hits, misses = 0, 0

    if PRETRAINED == "GLOVE_V1":
        path_to_glove_file = "../data/glove.6B.100d.txt"
        embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        print("In pretrained golve, found %s word vectors." % len(embeddings_index))
        embedding_dim = 100 

    ## WORD2VEC 
    else:
        import gensim.downloader

        if PRETRAINED == "word2vec":
            embeddings_index =  gensim.downloader.load('word2vec-google-news-300')
            
        if PRETRAINED == "glove_300":
            print("embedding golve_300 loading!!!")
            embeddings_index =  gensim.downloader.load('glove-wiki-gigaword-300')
        embedding_dim = len(embeddings_index["he"])

    # Prepare embedding matrix
    embedding_matrix = np.zeros((n_words, embedding_dim))
    for word, i in word_to_ix.items():
        if PRETRAINED == "GLOVE_V1":
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        else:
            try:
                embedding_matrix[i] = embedding_vector[word]
                hits += 1
            except:
                misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

# ---------- Model -----------


input_ = Input(shape=(max_len,))

#---with pre-train----
if PRETRAINED != "":
    model = Embedding(input_dim=n_words, output_dim=embedding_dim, 
                      embeddings_initializer=Constant(embedding_matrix), 
                      trainable=True, #since there are many missing words in the pretrained
                      input_length=max_len)(input_)
else:
    model = Embedding(input_dim=n_words, output_dim=embedding_dim, 
                      input_length=max_len)(input_)

model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
model = Model(input_, out)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/BiLSTM_testing',
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])


size_list = list(range(10000,51000,5000))
pbar = tqdm(total=len(size_list))
for i in size_list:
    pbar.update(1)
    #train with mini batch
    X_tr_batch = X_tr[0:i]
    y_tr_batch = y_tr[0:i]
    history = model.fit(X_tr_batch, y_tr_batch, batch_size=32, epochs=5, 
                     verbose=0, callbacks=[TqdmCallback(verbose=2)],
                     validation_data=(X_val, y_val))
    
    
    # Test for unseen
    unseen_pred = model.predict(X_te_unseen, verbose=1)
    test_labels = pred2labels(y_te_unseen).tolist()
    pred_labels = pred2labels(unseen_pred).tolist()
    report = classification_report(test_labels, pred_labels)
    training_size = "Training size:"+(str(len(X_tr_batch)))
    with open('BiLSTM_train.txt', "a") as file:
        file.write('\n')
        file.write(training_size)
        file.write('\n')
        file.write(report)
        file.write('\n')
    file.close()




