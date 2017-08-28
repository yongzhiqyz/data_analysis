# Multilayer Perceptron (MLP) for multi-class softmax classification:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras


# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
print('y_test shape:', y_test.shape)
x_pred = np.random.random((1, 20))

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


hist = model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
print('history: ',hist.history)
score = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_pred)

print('score:', score)
print ('metrics name: ', model.metrics_names)
print('y_pred: ', y_pred)

model.summary()


# save the weights
model.save_weights('weights.h5')
model.save('my_model.h5')

# save the trained model without weights
from keras.models import model_from_json
json_string = model.to_json()
# print (json_string)
import json
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

# print ('========================')
# with open('weights.json') as json_data:
#     d = json.load(json_data)
#     print (d)

