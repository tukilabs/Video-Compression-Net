from utils.network import SpyNetwork
from utils.basics import warp
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import math
import pickle as pkl


if __name__ == "__main__":
    filepath = "checkpoints/tfof.chk"
    checkpointFrequency = 25
    direc = "vimeo_septuplet/sequences/"
    subdircount = 0

    for item in os.listdir(direc):
        subdircount += 1

    starting = True

    spy = SpyNetwork()

    tfprvs = tf.placeholder(tf.float32, shape=[4, 256, 448, 3], name="first_frame")
    tfnext = tf.placeholder(tf.float32, shape=[4, 256, 448, 3], name="second_frame")
    tflow = spy(tfprvs, tfnext)
    recon = warp(tfprvs, tflow)
    cost = tf.reduce_mean(tf.squared_difference(tfnext, recon))
    train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

    tfvideo_batch = tf.get_variable("tfvideo_batch", initializer=tf.constant(0))
    increment_video_batch = tf.assign(tfvideo_batch, tfvideo_batch + 1)
    directory = tf.get_variable("directory", initializer=tf.constant(1))

    increment_directory = tf.assign(directory, directory + 1)
    init_video_batch_updater = tf.assign(tfvideo_batch, 0)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        with open("checkpoints/spymodeltf.pkl", "rb") as f:
            spy.set_weights(pkl.load(f))
        if starting:
            saver.restore(sess, filepath)

        load_dir = directory.eval()

        for i in range(load_dir, subdircount + 1):
            subdir = direc + str(i).zfill(5) + '/'
            subsubdircount = 0
            for item in os.listdir(subdir):
                subsubdircount += 1

            start_video_batch = tfvideo_batch.eval() if starting else 0
            num_video_batch = math.floor(subsubdircount / 4)
            starting = False

            for video_batch in range(start_video_batch, num_video_batch):
                for batch in range(1, 8):
                    bat = subdir + str(4 * video_batch + 1).zfill(4) + '/im' + str(batch) + '.png'
                    bat = np.array(Image.open(bat)).astype(np.float32) * (1.0 / 255.0)
                    bat = np.reshape(bat, (1, 256, 448, 3))
                    for item in range(2, 5):
                        img = subdir + str(4 * video_batch + item).zfill(4) + '/im' + str(batch) + '.png'
                        img = np.array(Image.open(img)).astype(np.float32) * (1.0 / 255.0)
                        img = np.reshape(img, (1, 256, 448, 3))
                        bat = np.concatenate((bat, img), axis=0)

                    if batch == 1:
                        prevReconstructed = bat

                    else:
                        loss, _ = sess.run([cost, train], feed_dict={tfprvs: prevReconstructed, tfnext: bat})
                        prevReconstructed = bat

                increment_video_batch.op.run()
                print("recon loss = {:.8f} video = {}, directory = {}".format(loss, video_batch, i))
                # print(tfvideo_batch.eval(), directory.eval())
                if video_batch % checkpointFrequency == 0:
                    weights = spy.get_weights()
                    pkl.dump(weights, open("checkpoints/spymodeltf.pkl", "wb"))
                    saver.save(sess, filepath)
