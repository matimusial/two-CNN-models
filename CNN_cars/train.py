from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from keras.layers import Dense



data = np.genfromtxt('car.data', delimiter=",", dtype=str)

X = data[:,:-1]
y = data[:,-1]

X_mapped = mapdata(X)

X_norm = normalize(X_mapped)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)    #zamiana na liczby

y_one_hot = to_categorical(y_encoded)   #zamiana na wektory

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_one_hot, test_size=0.2, random_state=15)


model = models.Sequential()
model.add(Dense(128, input_shape=(6,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

_, final_accuracy = model.evaluate(X_test,  y_test, verbose=2)

model.save(f"{round(final_accuracy, 3)}.h5")