# -*- coding: utf-8 -*-

#import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
#parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'zhou_00000.png'), help='path to the no_makeup image')
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup'), help='path to the no_makeup dir')
parser.add_argument('--makeuped', type=str, default=os.path.join('imgs', 'result'), help='path to the result dir')
parser.add_argument('--output', type=str, default=os.path.join('imgs', 'result'), help='path to the result dir')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
filelist = os.listdir(args.no_makeup)
non_makeups = []
for file in filelist:
    #path = os.path.join(args.no_makeup, file)
    non_makeups.append(file)




makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((img_size, img_size, 3))

tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

for i in range(len(makeups)):
    makeuppath = os.path.join(args.output)
    if not os.path.exists(makeuppath):
        os.mkdir(makeuppath)
    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    for j in range(len(non_makeups)):
        non_makeup = os.path.join(args.no_makeup,non_makeups[j])
        print('process nonmakeup:', non_makeup)
        no_makeup = cv2.resize(imread(non_makeup), (img_size, img_size))
        X_img = np.expand_dims(preprocess(no_makeup), 0)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = deprocess(Xs_)
        result[:img_size, :img_size, :] = Xs_[0]
        imsave(os.path.join(makeuppath, non_makeups[j]), result)

