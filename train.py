import tensorflow as tf
import numpy as np
from dataset import MyData
from utils import inverse_channel, decode_mask
from model import DeepLab
import math


is_training = tf.placeholder(dtype=tf.bool, shape=())
num_class = 21
batch_size = 10
init_lr = 0.001
power = 0.9
weight_decay = 0.0005
epoch_nums = 60
IMAGE_MEAN = [104.00698793, 116.66876762, 122.67891434]
train_examples_num = 8498
val_examples_num = 2857
train_step_per_epoch = math.ceil(train_examples_num/batch_size)
val_step_per_epoch = math.ceil(val_examples_num/batch_size)
max_iter = epoch_nums * train_step_per_epoch
print("Training's max_iter = %d" % max_iter)


data_dict = np.load("initial.npy", encoding="bytes").item()
print("Reading initial model")

m = DeepLab("data/dataset/train.record", "data/dataset/val.record", num_class, epoch_nums, train_step_per_epoch, val_step_per_epoch, weight_decay, power, init_lr, IMAGE_MEAN)
m.train(data_dict)
