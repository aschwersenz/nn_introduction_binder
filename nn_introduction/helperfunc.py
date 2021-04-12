from keras.datasets import mnist
import numpy as np


def print_network_info(network):
  weights, biases, activation, activation_prime = network
  
  print('Created a neural network with the following matrices: {:s}'.format(str([w.shape for w in weights])))
  print('As activation function it will use "{:s}" and its derivative "{:s}"'.format(activation.__name__, activation_prime.__name__))
  print('The network contains {:d} parameters!'.format(np.sum([w.shape[0] * w.shape[1] for w in weights]) + np.sum([b.shape[0] * b.shape[1] for b in biases])))


def print_epoch_info(epoch, error_val, accuracy_val, len_samples, duration):
  error_val = error_val / len_samples
  accuracy_val = (accuracy_val / len_samples) * 100
  
  print('Error / accuracy after epoch {:2d}: {:1.4f} / {:3.2f}% [Duration: {:4.2f}s]'.format(epoch + 1, error_val.flatten()[0], accuracy_val, duration))
  
  
def load_mnist():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  train_images = [(i.reshape(784, 1) / 255.0, np.eye(10)[j].reshape(10, 1)) for i, j in zip(x_train, y_train)]
  test_images = [(i.reshape(784, 1) / 255.0, np.eye(10)[j].reshape(10, 1)) for i, j in zip(x_test, y_test)]
  
  return train_images, test_images
  