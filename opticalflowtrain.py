from utils.network import SpyNetwork
from utils.basics import warp
import tensorflow.compat.v1 as tf
import os
import numpy as np
from PIL import Image
import math
import pickle as pkl
import argparse
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chkfile", "-c", default="checkpoints/oftrain.chk",
                        help="Checkpointing file\n"
                             "Default=`checkpoints/oftrain.chk`")

    parser.add_argument("--pklfile", "-p", default="checkpoints/oftrain.pkl",
                        help="Pkl file to save weights of the trained network\n"
                             "Default=`checkpoints/oftrain.pkl`")

    parser.add_argument("--input", "-i", default="vimeo_septuplet/sequences/",
                        help="Directory where training data lie. The structure of the directory should be like:\n"
                             "vimeo_septuplet/sequences/00001/\n"
                             "vimeo_septuplet/sequences/00002\n"
                             "...............................\n"
                             "For each vimeo_septuplet/sequences/x there should be subfolders like:\n"
                             "00001/0001\n"
                             "00001/002\n"
                             ".........\n"
                             "Check vimeo_septuplet folder. \n"
                             "Download dataset for more information. For other dataset, you can parse the input\n"
                             "in your own way\n"
                             "Default=`vimeo_septuplet/sequences/`")

    parser.add_argument("--frequency", "-f", type=int, default=25,
                        help="Number of steps to saving the checkpoints\n"
                             "Default=25")


    parser.add_argument("--restore", "-r", action="store_true",
                        help="Whether to restore the checkpoints to continue interrupted training, OR\n"
                             "Start training from the beginning")

    parseargs = parser.parse_args()
    return parseargs



if __name__ == "__main__":
    args = parse_args()
    filepath = "checkpoints/tfof.chk"
    direc = "vimeo_septuplet/sequences/"
    subdircount = 0

    for item in os.listdir(direc):
        subdircount += 1

    starting = args.restore

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
    init_directory_updater = tf.assign(directory, 1)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if starting:
            saver.restore(sess, args.chkfile)

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
                if video_batch % args.frequency == 0:
                    pkl.dump(spy.get_weights(), open(args.pklfile, "wb"))
                    saver.save(sess, args.chkfile)

                pkl.dump(spy.get_weights(), open(args.pklfile, "wb"))
                saver.save(sess, args.chkfile)

                init_video_batch_updater.op.run()
                increment_directory.op.run()

            init_directory_updater.op.run()

