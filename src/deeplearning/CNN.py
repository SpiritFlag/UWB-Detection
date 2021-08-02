from src.global_vars import *
from src.deeplearning.global_vars import *
from src.deeplearning.MyCallback import *

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class CNN(tf.keras.Model):
    def __init__(self):
        try:
            super(CNN, self).__init__()

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.model = self.buildModel()
            self.model.compile(loss=loss_function, optimizer=optimizer)
            self.model.summary()

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[CNN.__init__:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")

    def buildModel(self):
        try:
            self.input_layer = tf.keras.layers.Input(
                shape=(1, size_input_layer, size_input_filter), name="input")
            self.noise_layer = tf.keras.layers.GaussianNoise(
                stddev=0.1, name="gaussian_noise")

            if is_gaussian_noise is True:
                layer = self.noise_layer(self.input_layer)
            else:
                layer = self.input_layer

            for idx in range(len(size_filter)):
                conv_layer = tf.keras.layers.Conv2D(filters=size_filter[idx],
                    kernel_initializer='he_uniform',
                    kernel_size=(1, size_conv_layer), name="conv"+str(idx+1))
                layer = conv_layer(layer)

                if is_batch_normalization is True:
                    batch_layer = tf.keras.layers.BatchNormalization(
                        name="conv_batch"+str(idx+1))
                    layer = batch_layer(layer)

                activation_layer = tf.keras.layers.Activation(
                    tf.nn.relu, name="conv_activation"+str(idx+1))
                layer = activation_layer(layer)

                pool_layer = tf.keras.layers.MaxPooling2D(
                    pool_size=(1, size_pool_layer), name="pool"+str(idx+1))
                layer = pool_layer(layer)

            self.flatten_layer = tf.keras.layers.Flatten(name="flatten")
            layer = self.flatten_layer(layer)

            for idx in range(len(size_dense_layer)):
                dense_layer = tf.keras.layers.Dense(
                    units=size_dense_layer[idx], name="dense"+str(idx+1))
                layer = dense_layer(layer)

                if is_batch_normalization is True:
                    batch_layer = tf.keras.layers.BatchNormalization(
                        name="batch"+str(idx+1))
                    layer = batch_layer(layer)

                activation_layer = tf.keras.layers.Activation(
                    tf.nn.relu, name="activation"+str(idx+1))
                layer = activation_layer(layer)

                if dropout_rate > 0:
                    dropout_layer = tf.keras.layers.Dropout(
                        rate=dropout_rate, name="dropout"+str(idx+1))
                    layer = dropout_layer(layer)

            outputLayer = tf.keras.layers.Dense(
                units=size_output_layer, activation=tf.nn.softmax,
                name="output")
            layer = outputLayer(layer)

            return tf.keras.Model(self.input_layer, layer)

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[CNN.buildModel:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def trainModel(self, input, label, val_input, val_label, save_model=True):
        try:
      # if os.path.isdir(model_full_path) is False:
      #   os.mkdir(model_full_path)

      # n_slice = int(size_output_layer / size_slice)
            if len(input) % batchSize == 0:
                nBatch = int(len(input) / batchSize)
            else:
                nBatch = int(len(input) / batchSize) + 1

            # hist = self.model.fit(input, label, epochs=learning_epoch, batch_size=batchSize, verbose=1)
            hist = self.model.fit(input, label, epochs=learning_epoch, batch_size=batchSize, verbose=0,
            callbacks=[MyCallback(nBatch=nBatch, patience=patience, train_data=[input, label], validation_data=[val_input, val_label],\
              test_fnc=self.testModel)]\
            )

      # if save_model is True:
      #   self.model.save(model_full_path)

            return hist

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[CNN.trainModel:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")


  #
  # def restore_model(self, path):
  #   try:
  #     self.model = tf.keras.models.load_model(path, compile=False)
  #
  #   except Exception as ex:
  #     _, _, tb = sys.exc_info()
  #     print("[CNN.restore_model:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
  #
  #
  #

    def testModel(self, input, type=0):
        try:
            if type == 1:
                if len(input) % batchSize == 0:
                    nBatch = int(len(input) / batchSize)
                else:
                    nBatch = int(len(input) / batchSize) + 1
                return self.model.predict(input, batch_size=batchSize, callbacks=[MyCallback(nBatch=nBatch)])
            else:
                return self.model.predict(input, batch_size=batchSize)

        except Exception as ex:
          _, _, tb = sys.exc_info()
          print("[CNN.testModel:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
