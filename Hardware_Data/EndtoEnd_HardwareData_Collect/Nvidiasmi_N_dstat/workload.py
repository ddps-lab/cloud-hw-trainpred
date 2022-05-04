from datetime import datetime
import math
import time
import pickle
import argparse
import tensorflow as tf
import numpy as np
import dataset_info
import model_info

# GPU device check
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))

# Get arguments for job
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=224, type=int)
parser.add_argument('--model', default='VGG19', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--prof_point', default=10, type=float)
parser.add_argument('--prof_len', default=1, type=int)
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--instance_type', default='EC2', type=str)
args = parser.parse_args()
    
# Dataset Info
dataset = dataset_info.select_dataset(args.dataset)
model_name = args.model

num_classes = dataset['num_classes']
img_rows = dataset['img_rows']
img_cols = dataset['img_cols']
img_channels = dataset['img_channels']
num_data = dataset['num_data']
num_test = dataset['num_test']

batch_size = args.batch_size
prof_point = args.prof_point
batch_num = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_point)
# prof_start = math.floor(batch_num * prof_point)
# prof_len = args.prof_len
# prof_range = '{}, {}'.format(prof_start, prof_start + prof_len)
optimizer = 'SGD'

###################### Build Fake Dataset ######################
x_train_shape = (num_data, img_rows, img_cols, img_channels)
y_train_shape = (num_data, 1)

x_test_shape = (num_test, img_rows, img_cols, img_channels)
y_test_shape = (num_test, 1)

x_train = np.random.rand(*x_train_shape)
y_train = np.random.randint(num_classes, size=y_train_shape)
x_test = np.random.rand(*x_test_shape)
y_test = np.random.randint(num_classes, size=y_test_shape)
###############################################################
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Select model from model info module
model = model_info.select_model(model_name, input_shape, num_classes)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])


epoch_dict = {}
class epochTimeCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        print(epoch)
        global epoch_start
        self.epoch_time_start = time.time()
        epoch_start=datetime.fromtimestamp(self.epoch_time_start).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print('epoch_start : '+str(epoch_start))
        epoch_dict[epoch] = [epoch_start]
    def on_epoch_end(self, epoch, logs=None):
        global epoch_end
        self.epoch_time_end = time.time()
        epoch_end=datetime.fromtimestamp(self.epoch_time_end).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print('epoch_end : '+str(epoch_end))
        epoch_dict[epoch].append(epoch_end)


model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks = epochTimeCallback())

import pickle
epoch_ver_filename= './'+str(model_name)+'_batchsize'+str(batch_size)+'_datasize'+str(args.dataset)+'_total_epoch'+str(epochs)+"_totaldata"+str(num_data)+'.csv'           

# save data
with open(epoch_ver_filename,'wb') as fw:
    pickle.dump(epoch_dict, fw)
# load data
with open(epoch_ver_filename, 'rb') as fr:
    user_loaded = pickle.load(fr)
# show data
print(user_loaded)
