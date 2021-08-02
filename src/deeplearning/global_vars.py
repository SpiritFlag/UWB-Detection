from src.global_vars import *

signalPath = dataPathPrefix + "B_signal/"

is_gaussian_noise = True
dropout_rate = 0.2
is_batch_normalization = True

learning_rate = 1e-3
batchSize = 1024
patience = 5
learning_epoch = 1000

index_st = 0
index_ed = 1016
size_input_layer = int(index_ed - index_st)
size_input_filter = 2

size_filter = [32, 64, 128]
size_conv_layer = 3
size_pool_layer = 2
size_dense_layer = [2048, 1024]

loss_function = "categorical_crossentropy"
size_output_layer = classification
