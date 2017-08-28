# MLP for binary classification:
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))
x_pred = np.random.random((2,20))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=400,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_pred)
print('metrics name:', model.metrics_name, 'score: ', score)
print('x_pred:', x_pred)
print('y_pred:', y_pred)