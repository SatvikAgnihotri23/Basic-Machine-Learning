
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

'''
Loading Data in + formatting
'''

data = keras.datasets.fashion_mnist

# import training + testing data in 4 types of variables we need
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# creating an index so you know what each label in 'print(train_labels[6])' is
# these are the different classifications the network can make
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# dividing values by 255 to shrink scale of color classification - better to have smaller numbers - less computationally intense
train_images = train_images / 255.0
test_images = test_images / 255.0

# shows how data appears as numerical values
# print(train_images[7])
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

# flatten the data - take any interior array and make it more condense
# [[1,[2],[3]] --> [1,2,3]
# goal is to flatten data into one row of [ ] --> 784 (28x28) (input layer)

'''
building a model
'''

# defining the architecture of the model
# defining layers
model = keras.Sequential([
    # input layer is flattened to 728
    keras.layers.Flatten(input_shape=(28, 28)),
    # dense layer - 128 neurons w/ activation function of ReLu
    # dense layer = fully connected layer
    keras.layers.Dense(128, activation="relu"),
    # soft max picks values for each neuron such that they all add up to 1
    keras.layers.Dense(10, activation="softmax")
])

# setting up parameters for the model
# code = self explanatory
# google - what is an optimizer
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

'''
training the model

'''

# - randomly picks images and values to train on --> epoch = how many times the model will see the same information
# *reasoning is because the order in which the data comes in changes how the model sets up
# essentially epochs = same images in a dif order
# have to play with epochs value to find optimal fit

model.fit(train_images, train_labels, epochs=5)

'''
 using the model to predict
'''


# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)
# print(predictions[0]) # shows what an example of output currently is

# if you only want to predict one specific image
# predictions = model.predict([test_image[7]])
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual " + class_names[test_labels[i]])
    # takes prediction for 0 and indexing them into class names, then listing the first (maximum) prediction as a title
    plt.title("Predictions " + class_names[np.argmax(predictions[i])])
    plt.show()





