from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=1):
    model = keras.Sequential()
    model.add(layers.Dense(output_length, activation=activation_f, input_shape=(input_length,)))
    return model

def define_dense_model_with_hidden_layer(input_length, activation_func_array=['relu', 'sigmoid'], hidden_layer_size=10, output_length=1):
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation=activation_func_array[0], input_shape=(input_length,)))
    model.add(layers.Dense(output_length, activation=activation_func_array[1]))
    return model

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

def binarize_labels(labels, target_digit=2):
    labels = (labels == target_digit).astype(int)
    return labels

def fit_mnist_model_single_digit(x_train, y_train, target_digit, model, epochs=10, batch_size=128):
    y_train = binarize_labels(y_train, target_digit)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_mnist_model_single_digit(x_test, y_test, target_digit, model):
    y_test = binarize_labels(y_test, target_digit)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy
