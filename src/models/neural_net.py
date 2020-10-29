import pickle
import numpy as np
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

from sklearn.preprocessing import MultiLabelBinarizer


from sklearn.model_selection import train_test_split

with open('data/processed/target_train.pkl','rb') as f:
    Y_train=pickle.load(f)
print("Loaded the training target variable Y from data/processed/target_train.pkl.")

with open('data/processed/w2v_features_train.pkl','rb') as f:
    X_train=pickle.load(f)
print("Loaded X from data/processed/w2v_features_train.pkl.\n")

print("\tShape of X_train is {X_train}.\n".format(X_train=X_train.shape))

w2v_nn = Sequential([
    Dense(300, input_shape=(300,)),
    Activation('relu'),
    Dense(np.shape(Y_train)[1]),
    Activation('softmax'),
])

w2v_nn.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## needs apt-get install graphiz && pip install pydot
# from keras.utils import plot_model
# plot_model(w2v_nn, to_file='models/nn_figs/model.png', show_shapes=True)
# print("Saved visualization of model architecture to models/nn_figs/model.png.")

print("Starting training...")
history = w2v_nn.fit(X_train, Y_train, epochs=5000, batch_size=500,verbose=1) #5000
print("Training done!\n")

###
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
if keras.__version__ < '2.3.0':
    plt.plot(history.history['acc'])
else:
    plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('models/nn_figs/nn_training_validation_accuracy.png')

# Plot training & validation loss values
plt.clf()
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('models/nn_figs/nn_training_validation_loss.png')

print("Saved training history visualization to models/nn_figs/nn_training_validation_accuracy.png and models/nn_figs/nn_training_validation_loss.png.\n")

w2v_nn.save("models/classifier_nn.h5")
print("Saved the model to models/classifier_nn.h5.\n")