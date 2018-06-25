"""
@author - Sivaramakrishnan
"""
import tensorflow as tf
import numpy as np
import csv
from sklearn.model_selection import train_test_split

class data_loader(object):
    def __init__(self,LABELS_FILE,BATCH_SIZE = 5):
        self.LABELS_FILE = LABELS_FILE
        self.total_data = {}
        self.BATCH_SIZE = BATCH_SIZE

    def read_csv(self):
        total_data_ = {}
        with open(self.LABELS_FILE) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i,each_line in enumerate(csv_reader):
                total_data_[each_line[0]] = each_line[1].split("|")
        all_labels_list = []
        for each_label in total_data_.values():
            all_labels_list += each_label
        unique_labels = list(set(all_labels_list))
        np_array = np.eye(len(unique_labels))

        for i, each_image in enumerate(total_data_):
            label_ = np.zeros((len(unique_labels),))
            for each_label in total_data_[each_image]:
                label_ += np_array[unique_labels.index(each_label)]
            self.total_data[each_image] = label_

    def train_test_split(self):
        all_images = np.array(list(self.total_data.keys()))
        all_labels = np.array(list(self.total_data.values()),dtype=np.float32)

        self.train_data,self.val_data,self.train_labels,self.val_labels = train_test_split(all_images,all_labels)

    def parse_function(self,img_file_name,label):
        image = tf.read_file(img_file_name)
        image = tf.image.decode_image(image)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image_resized = tf.image.resize_image_with_crop_or_pad(image,224,224)
        return image_resized,label


    def train_data_loader(self):
        tf_train_data = tf.constant(self.train_data)
        tf_train_labels = tf.constant(self.train_labels)

        tf_val_data = tf.constant(self.val_data)
        tf_val_labels = tf.constant(self.val_labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_data,tf_train_labels))
        train_dataset = train_dataset.map(self.parse_function)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        val_dataset = tf.data.Dataset.from_tensor_slices((tf_val_data,tf_val_labels))
        val_dataset = val_dataset.map(self.parse_function)
        val_dataset = val_dataset.batch(self.BATCH_SIZE)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        next_images,next_labels = iterator.get_next()

        train_op = iterator.make_initializer(train_dataset)
        val_op = iterator.make_initializer(val_dataset)

        return [next_images,next_labels,train_op,val_op]