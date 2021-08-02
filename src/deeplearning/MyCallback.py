from src.global_vars import *
from src.deeplearning.global_vars import *

import os
import gc
import sys
import timeit
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



class MyCallback(tf.keras.callbacks.Callback):
  def __init__(self, nBatch=0, patience=0, train_data=[], validation_data=[], test_fnc=None, log_path=""):
    try:
      super(MyCallback, self).__init__()
      self.nBatch = nBatch
      self.patience = patience
      self.train_data = train_data
      self.validation_data = validation_data
      self.test_fnc = test_fnc

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.__init__:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_train_begin(self, epoch, logs=None):
    try:
      self.best_success = 0
      self.best_success2 = 0
      self.best_wait = 0
      self.best_epoch = 0
      self.best_weights = None

      self.n_val = len(self.validation_data[0])
      self.n_slice = len(self.validation_data[1])

      # self.fileL = open(self.log_path + "/loss", "w")
      # self.fileS = open(self.log_path + "/success", "w")

      print("\n\t*** TRAIN BEGIN ***", end="\n\n")

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_train_begin:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_epoch_begin(self, epoch, logs=None):
    try:
      print(f"Epoch {epoch+1}/{learning_epoch}")
      self.pbar = tqdm(total=self.nBatch, desc="TRAINING", ncols=100, unit=" batch")
      self.epoch_time = timeit.default_timer()

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_epoch_begin:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_train_batch_end(self, batch, logs=None):
    try:
      self.pbar.update(1)
      if batch+1 < self.nBatch:
        print(f" loss: {logs.get('loss'):.4f} - avg: {logs.get('loss')/self.n_slice:.4f}     ", end="\r")
      else:
        print("                              ", end="\r")

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_train_batch_end:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_epoch_end(self, epoch, logs=None):
    try:
      self.pbar.close()

      train_res = self.test_fnc(self.train_data[0], type=1)
      train_ans = self.train_data[1]

      success = 0
      for x in range(len(train_res)):
        if np.argmax(train_res[x]) == np.argmax(train_ans[x]):
          success += 1


      val_res = self.test_fnc(self.validation_data[0], type=1)
      val_ans = self.validation_data[1]


      success2 = 0
      for x in range(len(val_res)):
        if np.argmax(val_res[x]) == np.argmax(val_ans[x]):
          success2 += 1


      # self.fileL.write(str(logs.get('loss')) + "\t")
      # self.fileS.write(str(success) + "\t")

      print(f"\t\tLOSS= {logs.get('loss'):.4f}")
      print(f"\t\tTRAIN\t\tSUCCESS=\t{success:5d} / {len(self.train_data[0]):5d} ({100*success/len(self.train_data[0]):2.2f}%)")
      print(f"\t\tVALIDATION\tSUCCESS=\t{success2:5d} / {self.n_val:5d} ({100*success2/self.n_val:2.2f}%)")

      if success2 != 0:
        if success2 > self.best_success:
          self.best_wait = 0
          self.best_success = success2
          self.best_success2 = success
          self.best_epoch = epoch + 1

          print("\tSTORING MODEL WEIGHTS..")
          self.best_weights = self.model.get_weights()

        elif success/len(self.train_data[0]) < 0.95:
          self.best_wait = 0
          print("\tWAITING UNTIL THE TRAIN SUCCESS RATIO REACHES 95%..")

        else:
          self.best_wait += 1
          if self.best_wait >= self.patience:
            self.model.stop_training = True

          print(f"\t{self.best_wait} / {self.patience}\tWAITING PATIENCE..", end="\t")
          print(f"{self.best_success:5d} / {self.n_val:5d} ({100*self.best_success/self.n_val:2.2f}%)")

      gc.collect()
      print(f"\t\tEPOCH TIME= {timeit.default_timer()-self.epoch_time:.3f} (sec)")
      print("\n\n", end="")

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_epoch_end:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_train_end(self, epoch, logs=None):
    try:
      print(f"\t*** Early stopping and restoring model weights from the epoch {self.best_epoch}. ***")
      self.model.set_weights(self.best_weights)

      train_res = self.test_fnc(self.train_data[0], type=1)
      train_ans = self.train_data[1]

      tsne = TSNE(random_state = 42).fit_transform(train_res)
      for x in range(len(train_res)):
          plt.plot(tsne[x][0], tsne[x][1])
          plt.text(tsne[x][0], tsne[x][1], str(np.argmax(train_ans[x])+1),
              color = colors[np.argmax(train_ans[x])],
              fontdict = {'weight':'bold','size':9})
      plt.savefig("train.png", dpi=300)
      plt.close()

      print(f"\t\tTRAIN\t\tSUCCESS=\t{self.best_success2:5d} / {len(self.train_data[0]):5d} ({100*self.best_success2/len(self.train_data[0]):2.2f}%)")
      print(f"\t\tVALIDATION SUCCESS=\t{self.best_success:5d} / {self.n_val:5d} ({100*self.best_success/self.n_val:2.2f}%)")
      print("\n\n", end="")

      # self.fileL.close()
      # self.fileS.write("\n" + str(self.best_epoch))
      # self.fileS.close()

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_train_end:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_predict_begin(self, logs=None):
    try:
      self.pbar_predict = tqdm(total=self.nBatch, desc="PREDICTING", ncols=100, unit=" batch")

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_predict_begin:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_predict_batch_end(self, batch, logs=None):
    try:
      self.pbar_predict.update(1)

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_predict_batch_end:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")



  def on_predict_end(self, logs=None):
    try:
      self.pbar_predict.close()

    except Exception as ex:
      _, _, tb = sys.exc_info()
      print("[MyCallback.on_predict_end:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
