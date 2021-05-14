from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

__name__ = 'autoencoders'


def create_net(l1_regularization):
    # The regularization:
    regularizer = l1(0.001) if l1_regularization else None
    
    # The encoder:
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same',
         activity_regularizer=regularizer)(x)
    encoder = Model(input_img, encoded)

    # Features' shape is (4,4,8), i.e. 128-dimensional.

    # The decoder layers:
    decoder_input = Conv2D(8, (3, 3), activation='relu', padding='same')
    decoder_up01 = UpSampling2D((2, 2))
    decoder_conv01 = Conv2D(8, (3, 3), activation='relu', padding='same')
    decoder_up02 = UpSampling2D((2, 2))
    decoder_conv02 = Conv2D(16, (3, 3), activation='relu')
    decoder_up03 = UpSampling2D((2, 2))
    decoder_conv03 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    def decoder_from(enc_in):
        x = decoder_input(enc_in)
        x = decoder_up01(x)
        x = decoder_conv01(x)
        x = decoder_up02(x)
        x = decoder_conv02(x)
        x = decoder_up03(x)
        return decoder_conv03(x)

    # The decoder:
    encoded_img = Input(shape=(4,4,8))
    decoded_test = decoder_from(encoded_img)
    decoder = Model(encoded_img,decoded_test)

    # The whole autoencoder:
    decoded = decoder_from(encoded)
    autoencoder = Model(input_img, decoded)

    # The optimization:
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Returning all:
    return autoencoder, encoder, decoder


def prepare_data(data, corrupt=False):
    data = data.astype('float32') / 255.
    return np.reshape(data, (len(data), 28, 28, 1))


def corrupt(data):
    return data + np.random.normal(loc=0.0, scale=.2, size=data.shape) 


def load_data(must_corrupt=False):
    (y_train, _), (y_test, _) = mnist.load_data()
    
    y_train = prepare_data(y_train)
    y_test = prepare_data(y_test)
    
    x_train, x_test = y_train, y_test
    
    if must_corrupt:
        x_train = corrupt(x_train)
        x_test = corrupt(x_test)
        
    return x_train, y_train, x_test, y_test


def train(net, train_data, validation_data, epochs=30, batch_size=64, shuffle=True, **kwargs):
    net.fit(
        x=train_data[0],
        y=train_data[1],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_data=validation_data,
        **kwargs)
    

def imshowcompare(imgs1, imgs2, n=10):
    plt.figure(figsize=(n*2, 4))
    for c in range(n):
        ax = plt.subplot(2, n, c + 1)
        plt.imshow(imgs1[c].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, c + 1 + n)
        plt.imshow(imgs2[c].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    
def plotfeatures(encoded_images, n=10):
    plt.figure(figsize=(n*2, 8))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(encoded_images[i].reshape(4, 4 * 8).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    