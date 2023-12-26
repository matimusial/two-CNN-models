import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functions import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


X_train, y_train, X_test, y_test = read_data()

X_train_transposed, X_test_transposed = makearray(X_train, X_test)

print(y_train)

# model = Sequential()
# model.add(Conv2D(32, (4, 4), input_shape=(32, 32, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (4, 4), input_shape=(32, 32, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train_transposed, y_train, epochs=5, validation_data=(X_test_transposed, y_test))
#
# _, final_accuracy = model.evaluate(X_test_transposed,  y_test, verbose=2)