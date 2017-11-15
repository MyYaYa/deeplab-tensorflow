import tensorflow as tf
import numpy as np


class MyData(object):
    def __init__(self, record, image_mean, shuffle=False, buffer_size=1000, batch_size=10, repeat=False, repeat_times=None):
        self.record = record
        self.image_mean = image_mean
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.repeat = repeat
        self.repeat_times = repeat_times


    def preprocess_function(self, example_proto):
        features = tf.parse_single_example(example_proto, features={
            'height' : tf.FixedLenFeature([], tf.int64),
            'width'  : tf.FixedLenFeature([], tf.int64),
            'image_raw' : tf.FixedLenFeature([], tf.string),
            'label_raw' : tf.FixedLenFeature([], tf.string),
        })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        label = tf.decode_raw(features['label_raw'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        label_shape = tf.stack([height, width, 1])

        image = tf.reshape(image, image_shape)
        label = tf.reshape(label, label_shape)
        crop_size = tf.constant(321, tf.int32)

        def subtract(a, b):
            return tf.subtract(a, b)

        def zero():
            return tf.constant(0, tf.int32)

        padding_h = tf.cond(pred=tf.greater_equal(subtract(crop_size, height), tf.constant(0, tf.int32)),
                            true_fn=lambda: subtract(crop_size, height),
                            false_fn=lambda: zero())

        padding_w = tf.cond(pred=tf.greater_equal(subtract(crop_size, width), tf.constant(0, tf.int32)),
                            true_fn=lambda: subtract(crop_size, width),
                            false_fn=lambda: zero())
        image = tf.pad(image, paddings=[[0, padding_h], [0, padding_w], [0, 0]])
        label = tf.pad(label, paddings=[[0, padding_h], [0, padding_w], [0, 0]])
        image = tf.reshape(image, tf.stack([height + padding_h, width + padding_w, 3]))
        label = tf.reshape(label, tf.stack([height + padding_h, width + padding_w, 1]))
        img_seg = tf.concat([image, label], 2)

        croped_img_seg = tf.random_crop(img_seg, [crop_size, crop_size, 4])

        mirrored_img_seg = tf.image.random_flip_left_right(croped_img_seg)
        image = tf.slice(mirrored_img_seg, [0, 0, 0], [321, 321, 3])
        label = tf.slice(mirrored_img_seg, [0, 0, 3], [321, 321, 1])

        # subtract three channals‘ mean value
        image = tf.cast(image, dtype=tf.float32)
        mean = tf.constant(self.image_mean, dtype=tf.float32)
        mean = tf.reshape(mean, [1, 1, -1])
        image = tf.subtract(image, mean)
        # image = tf.div(image, 255)
        label = tf.cast(label, dtype=tf.int32)
        return image, label


    def build_dataset(self):
        dataset = tf.contrib.data.TFRecordDataset(self.record)
        dataset = dataset.map(self.preprocess_function)
        # shuffle
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        if self.repeat:
            dataset = dataset.repeat()
        if self.repeat_times is not None:
            dataset = dataset.repeat(self.repeat_times)
        dataset = dataset.batch(self.batch_size)
        #iterator = dataset.make_initializable_iterator()
        return dataset
