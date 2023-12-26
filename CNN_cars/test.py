import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functions import *
from keras.models import load_model

data = np.genfromtxt('CNN_cars/car.data', delimiter=",", dtype=str)

test = np.array(['low','low','5more','more','med','high'])

X = data[:,:-1]

y = np.unique(data[:,-1])

test_1 = np.vstack((X, test))

test_map = mapdata(test_1)

test_norm = normalize(test_map)

test = test_norm[-1,:].reshape(1,-1)


model = load_model('CNN_cars/0.945.h5')

predictions = model.predict(test)

print(predictions)

print(y)
print(f"Wynik predykcji: {y[np.argmax(predictions)]}")

