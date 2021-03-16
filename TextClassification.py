import tensorflow as tf
from tensorflow import keras
import numpy

data = keras.datasets.imdb

# Loading in the data + only take the top 10000 words that occur most frequently in the data set
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])
# what is printed is integer encoded words (the integer 1 stands for sum, 14 stands for sum, etc. nice for computer,
# not for us. Create a word index for it. Usually you have to do manual but tensorflow has one for this dataset

# data= imdb
word_index = data.get_word_index()

# k v = key value. Key=word. Value=integer
# take all the possible values from the orriginal data set and add 3 such that 1,2,3 no longer have assigned values
# assign 1 to start, 2 to unknown, and 3 to unused --> if data points are not valid we can assign them to this

word_index = {k: (v + 3) for k, v in word_index.items()}
# 'PAD tag' --> add zero into movie review list so that we can make each movie review the same length.
# Say one list has 100 data points and one list has 200 data points. Add 100 pad tags so that both have 200 data points.

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# swap all the value and key so that the dictionary is value then key. Integers point to words not the other way around
reverse_word_index = reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# preprocessing data - easy cuz keras does it for us
# takes training data --> adds the string PAD to any blank spaces onto 'post' (the end). It cuts any data past 250 words
train_date = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# proof that data has dif length for each entry
# print(len(train_data), len(test_data))

# decode all the data so it is human readable text
# try to get index i and if you cant recover a value - print a question mark
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print any sort of data to check the program is running the way we want it to
# print(decode_review(test_data[0]))

'''
Model Here
'''
model = keras.Sequential()
# layers to the model
'''
-  Embed layer tries to group similar word closer together - uses word vectors
- 10,000 word vectors- each one represents one word
- embed layer takes the vectors for whatever the input is and pass it on to the next layer
- Essentially: wants to take word vectors for similar words and put them close together (calculated by angle of 
separation between the vectors. Groups words how similar/different they are based on words surrounding them. 
- Output is 16 dimensions (coefficients to the vectors)
'''
model.add(keras.Embedding(10000, 16))

'''
Takes the dimensions output from the first layer (input to this layer) and puts it in a lower dimension
'''
model.add(keras.GlobalAveragePooling1D())
# dense layers
'''

'''
model.add(keras.Dense(16, activation="relu"))
model.add(keras.Dense(1, activation="sigmoid"))
# final output --> is the review good or bad? (0 or 1)

model.summary()

