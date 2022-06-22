from pathlib import Path
from glob import glob
import numpy as np
from os.path import join


def _find_images(data_path):
    paths = glob(join(data_path, '*.jpg'))
    paths.extend(glob(join(data_path, '*.jpeg')))
    paths.extend(glob(join(data_path, '*.png')))
    print("data_path: ", data_path)
    return paths


def image_paths(dir_path):
    """Find all filepaths for images in dir_path."""
    return _find_images(dir_path)

def deserialize_data(raw_record, raw=True, shape=[128, 128, 1]):
    """Deserialize single tfrecord."""
    import tensorflow as tf
    IMAGE_FEATURE_DESCRIPTION = {
        'image': tf.io.FixedLenFeature(shape, tf.float32),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'shape': tf.io.FixedLenFeature((3), tf.int64),
    }

    example = tf.io.parse_single_example(raw_record, IMAGE_FEATURE_DESCRIPTION)

    image = example["image"]
    image = tf.reshape(image, shape=example["shape"])
    label = example["label"]

    return (image, label)


def serialize_data(data):
    """Serialize single tfrecord."""
    import tensorflow as tf

    image, label = data
    feature = {
        "image": tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten().tolist())),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()
