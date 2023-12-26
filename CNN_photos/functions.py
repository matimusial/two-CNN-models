import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def prepareimageforpredict(filepath):
    image = Image.open(filepath)
    image1 = np.array(image)
    if image1.shape[0] != image1.shape[1]:
        print("Image does not have the same dimensions")
        return
    if image1.shape[0] == 32 and image1.shape[1] == 32:
        return image1/255
    else:
        img_resized = image.resize((32, 32))    #(szerokosc, wysokosc)
        img_resized = np.reshape(img_resized, (1, 32, 32, 3))
        return np.array(img_resized)/255    # w cnn (wysokosc, szerokosc)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def viewSingleTrainPicture(number):
    batches_meta = unpickle('CNN_photos/cifar-10-batches-py/batches.meta')
    batches_meta = batches_meta[b'label_names']
    X_train , y_train,_,_= read_data()
    plt.imshow(np.transpose(np.reshape(X_train[number], (3, 32, 32)), (1, 2, 0)))
    labelindex = y_train[number]
    plt.title(batches_meta[labelindex].decode('utf-8'))
    plt.show()

def read_data():
    batches_meta = unpickle('CNN_photos/cifar-10-batches-py/batches.meta')
    data_batch_1 = unpickle('CNN_photos/cifar-10-batches-py/data_batch_1')
    data_batch_2 = unpickle('CNN_photos/cifar-10-batches-py/data_batch_2')
    data_batch_3 = unpickle('CNN_photos/cifar-10-batches-py/data_batch_3')
    data_batch_4 = unpickle('CNN_photos/cifar-10-batches-py/data_batch_4')
    data_batch_5 = unpickle('CNN_photos/cifar-10-batches-py/data_batch_5')
    test_batch = unpickle('CNN_photos/cifar-10-batches-py/test_batch')
    # dict_keys([b'num_cases_per_batch', b'label_names', b'num_vis']) dla meta
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames']) dla data
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames']) dla test
    data1 = data_batch_1[b'data']
    data2 = data_batch_2[b'data']
    data3 = data_batch_3[b'data']
    data4 = data_batch_4[b'data']
    data5 = data_batch_5[b'data']
    X_test = test_batch[b'data']
    label1 = data_batch_1[b'labels']
    label2 = data_batch_2[b'labels']
    label3 = data_batch_3[b'labels']
    label4 = data_batch_4[b'labels']
    label5 = data_batch_5[b'labels']
    y_test = test_batch[b'labels']

    X_train = np.concatenate((data1, data2, data3, data4, data5), axis=0)
    y_train = np.concatenate((label1, label2, label3, label4, label5), axis=0)

    return X_train, y_train, np.array(X_test), np.array(y_test)

def makearray(X_train, X_test):
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0],3, 32, 32)) #rozmiar 3072 = 3 * 32 * 32  (numer zdjecia, kolor, wysokosc, szerokosc)
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0],3, 32, 32))
    X_train_transposed = np.transpose(X_train_reshaped, (0,2,3,1)) #0 - numer zdjecia, 2 - wysokosc, 3 - szerokosc, 1 - kolor
    X_test_transposed = np.transpose(X_test_reshaped, (0,2,3,1))

    return X_train_transposed/255, X_test_transposed/255


# Binary Crossentropy (binary_crossentropy):
# Zastosowanie: Problemy binarnej klasyfikacji, gdzie istnieje tylko dwie klasy.
# Przykład: Rozpoznawanie obrazów z jednym obiektem/klasą.

# Categorical Crossentropy (categorical_crossentropy):
# Zastosowanie: Problemy wieloklasowej klasyfikacji, gdzie obserwacje należą do jednej z wielu klas.
# Przykład: Rozpoznawanie ręcznie pisanego tekstu na obrazach (gdzie każda litera to osobna klasa).

# Sparse Categorical Crossentropy (sparse_categorical_crossentropy):
# Zastosowanie: Podobnie jak categorical_crossentropy, ale gdy etykiety są podane jako indeksy klas, a nie w formie kodowania one-hot.
# Przykład: Klasyfikacja obrazów, gdzie etykieta to liczba od 0 do liczby klas-1.