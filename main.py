# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import tensorflow as tf
from tensorflow import keras

def structure_string(list):
    # print(list)
    str = ""
    for i in range(len(list)):
        str = str + list[i]
    # print(str[:-1])
    return str[:-1] # get rid of /n

def load_dataset(dataset, lst, label):
    dataset.readline() # read title
    line = dataset.readline()
    while line != '':
        # print(line, end='')
        line_list = line.split(",")
        label.append(line_list[1])
        lst.append(structure_string(line_list[2:]))
        line = dataset.readline()

dataset = []
dataset_label = []
class_names = [0, 1]

filename = "train.csv"
raw_dataset = open(filename, 'r')
load_dataset(raw_dataset, dataset, dataset_label)
raw_dataset.close()

# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(dataset)
rng = np.random.RandomState(seed)
rng.shuffle(dataset_label)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(dataset))
train_dataset = dataset[:-num_validation_samples]
test_dataset = dataset[-num_validation_samples:]
train_label = dataset_label[:-num_validation_samples]
test_label = dataset_label[-num_validation_samples:]



#for i in range(len(train_dataset)):
#    print(train_label[i] + ": " + train_dataset[i])

print("Classes:", class_names)
print("Number of samples:", len(train_dataset))

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(train_dataset).batch(128)
vectorizer.adapt(text_ds)

#print(vectorizer.get_vocabulary()[:5])
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))
#output = vectorizer([["the cat sat on the mat"]])
#print(output.numpy()[0, :6])
#test = ["the", "cat", "sat", "on", "the", "mat"]
#print([word_index[w] for w in test])

# make a dict mapping words (strings) to their NumPy vector representation:
glove_file_path = "glove.6B.100d.txt"
embeddings_index = {}
with open(glove_file_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

# a corresponding embedding matrix that we can use in a Keras Embedding layer.
# It's a simple NumPy matrix where entry at index i is the pre-trained vector
# for the word of index i in our vectorizer's vocabulary.

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# load the pre-trained word embeddings matrix into an Embedding layer.
# Note that we set trainable=False so as to keep the embeddings fixed
# (we don't want to update them during training).

from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

# A simple 1D convnet with global max pooling and a classifier at the end.
from tensorflow.keras import layers

int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
model.summary()

# convert our list-of-strings data to NumPy arrays of integer indices. The arrays are right-padded.

x_train = vectorizer(np.array([[s] for s in train_dataset])).numpy()
x_test = vectorizer(np.array([[s] for s in test_dataset])).numpy()
#for i in range(len(x_test)):
#    print(x_test[i])
y_train = np.array(train_label)
y_test = np.array(test_label)
# categorical crossentropy as our loss since we're doing softmax classification.
# Moreover, we use sparse_categorical_crossentropy since our labels are integers.
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)
model.fit(x_train,
          y_train,
          batch_size=128, epochs=20, validation_data=(x_test, y_test))

print("Done")


