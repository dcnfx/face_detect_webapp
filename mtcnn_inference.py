#!usr/bin/env python3
# -*- coding:utf-8 _*-
import os
import numpy as np
import warnings
import tensorflow as tf
from models.mtcnn import detect_face
import cv2
warnings.filterwarnings("ignore")


class MtCnn:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session()
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
        self.minsize = 50  # minimum size of face
        self.threshold = [0.6, 0.7, 0.9]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin_ratio = 0.525

    def cut_face_norote(self, img):
        bounding_boxes, kpoint = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                         self.threshold, self.factor)
        if bounding_boxes.shape[0] > 0:
            y_min_cut, y_max_cut = bounding_boxes[0][[1, 3]].astype(np.int).tolist()
            x_min_cut, x_max_cut = bounding_boxes[0][[0, 2]].astype(np.int).tolist()
            y_min_cut = np.max([0, y_min_cut])
            x_min_cut = np.max([0, x_min_cut])
            y_max_cut = np.min([img.shape[0], y_max_cut])
            x_max_cut = np.min([img.shape[1], x_max_cut])
            cropped = img[y_min_cut:y_max_cut, x_min_cut:x_max_cut, :]
            scaled = cv2.resize(cropped, (112, 112))
        else:
            return None
        return scaled

    def predict(self, img):
        bounding_boxes, kpoint = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                         self.threshold, self.factor)
        return bounding_boxes, kpoint

