import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functions import *
from keras.models import load_model
from PIL import Image

imagepath = 'CNN_photos/photos/ship2.jpg'

image = prepareimageforpredict(imagepath)

plt.imshow(Image.open(imagepath))
plt.show()

batches_meta = unpickle('CNN_photos/cifar-10-batches-py/batches.meta')
batches_meta = batches_meta[b'label_names']
y=np.array([])
for i in range(len(batches_meta)):
    y = np.append(y,batches_meta[i].decode('utf-8')).astype(np.str_)

model = load_model('CNN_photos/0.859.h5')

predictions = model.predict(image)


print(f"Wynik predykcji: {y[np.argmax(predictions)]} dla prawdopodobienstwa {np.round(np.max(predictions),4)}")