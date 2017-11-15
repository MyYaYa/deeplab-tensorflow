import tensorflow as tf
import cv2
import os
import argparse

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_path, is_train):
    """Converts a dataset to tfrecords."""
    train_list = os.path.join(data_path, "train.txt")
    val_list = os.path.join(data_path, "val.txt")

    if is_train:
        filename = os.path.join(data_path, "train.record")
        image_list = train_list
    else:
        filename = os.path.join(data_path, "val.record")
        image_list = val_list

    writer = tf.python_io.TFRecordWriter(filename)

    with open(image_list, 'r') as f:
        length = len(f.readlines())
        f.seek(0)
        for i in range(length):
            name = f.readline().strip()

            # read data
            image = cv2.imread(os.path.join(data_path, "img/" + name + ".jpg"))
            label = cv2.imread(os.path.join(data_path, "cls_png/" + name + ".png"), cv2.IMREAD_GRAYSCALE)

            rows = image.shape[0]
            cols = image.shape[1]

            # convert image to raw string data
            image_raw = image.tostring()
            label_raw = label.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'image_raw': _bytes_feature(image_raw),
                'label_raw': _bytes_feature(label_raw)
            }))
            writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory to pascal voc aug data'
    )
    parser.add_argument(
        '--is_train',
        action='store_true',
        default=False,
        help='convert train data'
    )
    FLAGS, unparsed = parser.parse_known_args()
    convert_to(FLAGS.directory, FLAGS.is_train)