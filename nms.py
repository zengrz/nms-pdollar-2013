# File: nms.py
# Author: Zeng Ruizi (Rey)

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as pp

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def build_orientation_map(pb):
  pyramid_kernel_x = np.array([
    [0, 0, 1, 2, 3, 2, 1, 0, 0],
    [0, 1, 2, 3, 4, 3, 2, 1, 0],
    [1, 2, 3, 4, 5, 4, 3, 2, 1],
    [0, 1, 2, 3, 4, 3, 2, 1, 0],
    [0, 0, 1, 2, 3, 2, 1, 0, 0]
    ], np.float)
  pyramid_kernel_y = pyramid_kernel_x.transpose()

  preprocessing_kernel_y = tf.constant(pyramid_kernel_y, tf.float32, (9, 5, 1, 1), "pyramid_kernel_y")
  preprocessing_kernel_x = tf.constant(pyramid_kernel_x, tf.float32, (5, 9, 1, 1), "pyramid_kernel_x")

  pb4d = tf.expand_dims(pb, -1, "pb4d")
  pb_filtered_y = tf.nn.conv2d(pb4d, preprocessing_kernel_y, (1, 1, 1, 1), "SAME", name="pb_filtered_y")
  pb_filtered = tf.nn.conv2d(pb_filtered_y, preprocessing_kernel_x, (1, 1, 1, 1), "SAME", name="pb_filtered")

  gradient_kernel_y = tf.constant([-.5, 0, .5], tf.float32, (3, 1, 1, 1), "gradient_kernel_y")
  gradient_kernel_x = tf.constant([-.5, 0, .5], tf.float32, (1, 3, 1, 1), "gradient_kernel_x")

  padding_y = np.array([0, 0, 1, 1, 0, 0, 0, 0]).reshape(4, 2)
  padding_x = np.array([0, 0, 0, 0, 1, 1, 0, 0]).reshape(4, 2)

  pb_filtered_padded_y = tf.pad(pb_filtered, padding_y, "SYMMETRIC", "pb_filtered_padded_y")
  Oy = tf.nn.conv2d(pb_filtered_padded_y, gradient_kernel_y, (1, 1, 1, 1), "VALID", name="Oy")
  pb_filtered_padded_x = tf.pad(pb_filtered, padding_x, "SYMMETRIC", "pb_filtered_padded_x")
  Ox = tf.nn.conv2d(pb_filtered_padded_x, gradient_kernel_x, (1, 1, 1, 1), "VALID", name="Ox")

  Ox_padded_y = tf.pad(Ox, padding_y, "SYMMETRIC", "Ox_padded_y")
  Oyx = tf.nn.conv2d(Ox_padded_y, gradient_kernel_y, (1, 1, 1, 1), "VALID", name="Oyx")
  Ox_padded_x = tf.pad(Ox, padding_x, "SYMMETRIC", "Ox_padded_x")
  Oxx = tf.nn.conv2d(Ox_padded_x, gradient_kernel_x, (1, 1, 1, 1), "VALID", name="Oxx")

  Oy_padded_y = tf.pad(Oy, padding_y, "SYMMETRIC", "Oy_padded_y")
  Oyy = tf.nn.conv2d(Oy_padded_y, gradient_kernel_y, (1, 1, 1, 1), "VALID", name="Oyy")
  Oy_padded_x = tf.pad(Oy, padding_x, "SYMMETRIC", "Oy_padded_x")
  Oxy = tf.nn.conv2d(Oy_padded_x, gradient_kernel_x, (1, 1, 1, 1), "VALID", name="Oxy")

  Q = tf.atan(tf.divide(tf.multiply(Oyy, tf.sign(tf.negative(Oxy))), tf.add(Oxx, tf.constant(1e-5))), name="Q")
  pi = tf.constant(np.pi, tf.float32, name="pi")
  O = tf.mod(Q, pi, "O")

  return O

r = [-1, 1]
h, w = 1024, 1024
m = 1.01

pb_path = ""

pb_val = cv2.imread(pb_path, cv2.IMREAD_GRAYSCALE)
pb_val = pb_val[:h, :w]

pb = tf.constant(pb_val, dtype=tf.float32, shape=(1, h, w), name='pb')

O = build_orientation_map(pb) # 1, h, w, 1

pb = tf.expand_dims(pb, axis=-1, name='pb_expanded')

cos_O = tf.cos(O)
sin_O = tf.sin(O)

x_val = np.array(range(w))
y_val = np.array(range(h))

x_val = np.tile(x_val, h)
y_val = np.repeat(y_val, w)

x = tf.constant(x_val, dtype=tf.float32, shape=(1, h, w, 1), name='x')
y = tf.constant(y_val, dtype=tf.float32, shape=(1, h, w, 1), name='y')


for d in r:
  x_intp = x + d*cos_O
  y_intp = y + d*sin_O
  warp_coords = tf.concat([x_intp, y_intp], -1, name='coords')
  pb_resampled = tf.contrib.resampler.resampler(pb, warp_coords, name='pb_resampled')
  pb_mask = tf.cast(tf.logical_not(pb*m < pb_resampled), tf.float32, name='pb_mask')
  pb = pb * pb_mask

with tf.Session() as sess:
  pb_nms = sess.run(pb)

cv2.imwrite("pb_nms.png", np.squeeze(pb_nms))

