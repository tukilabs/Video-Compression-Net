import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from utils.network import VideoCompressor
import numpy as np
from PIL import Image
import pickle as pkl
from utils.basics import write_png
import argparse
import os
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="checkpoints/alloptimizedvideocomp.pkl",
                        help="Saved model that you want to decompress with. Should be same"
                             "as the model used in compression step for better reconstruction")

    parser.add_argument("--input", "-i", default="encoded/",
                        help="Directory where compressed files lie and what you want to decompress")

    parser.add_argument("--output", "-o", default="reconstructed/",
                        help="Directory where you want the reconstructed frames to be saved"
                             "Warning: Output directory might have previously reconstructed frames"
                             "which might be deceived as currently reconstructed frames")

    parser.add_argument("--frequency", "-f", type=int, default=7,
                        help="Should be same as that of compressor")


    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()
    w, h, _ = np.array(Image.open(args.input + "1.png")).shape
    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")

    compflow = tf.placeholder(tf.string, [1], name="compressed_of_string")
    cfx_shape = tf.placeholder(tf.int32, [2], name="compressed_of_lengthx")
    cfy_shape = tf.placeholder(tf.int32, [2], name="compressed_of_lengthy")

    compres = tf.placeholder(tf.string, [1], name="compressed_residue_string")
    rex_shape = tf.placeholder(tf.int32, [2], name="compressed_residue_lengthx")
    rey_shape = tf.placeholder(tf.int32, [2], name="compressed_residue_lengthy")

    _, _, _ = testnet(testtfprvs, testtfprvs)

    recimage = testnet.decompress(testtfprvs, compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape)

    testinit = tf.global_variables_initializer()

    num_frames = 0
    for item in os.listdir(args.input):
        if ".png" in item:
            num_frames += 1
        elif "of" in item:
            num_frames += 1

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        batch_range = args.frequency + 1
        for i in range(math.ceil(num_frames/args.frequency)):
            tenFirst = np.array(Image.open(args.input + str(i * args.frequency + 1) + '.png')).astype(np.float32) * (1.0 / 255.0)
            tenFirst = np.expand_dims(tenFirst, axis=0)
            sess.run(write_png(args.output + str(i * args.frequency +1) + ".png", tenFirst))

            if i == math.ceil(num_frames/args.frequency) -1 and num_frames % args.frequency != 0:
                batch_range = num_frames % args.frequency + 1

            for batch in range(2, batch_range):
                with open(args.input + "/of" + str(i * args.frequency + batch - 1) + ".vcn", "rb") as f:
                    flowpacked = tfc.PackedTensors(f.read())
                with open(args.input + "res" + str(i * args.frequency + batch - 1) + ".vcn", "rb") as f:
                    respacked = tfc.PackedTensors(f.read())

                flowtensors = [compflow, cfx_shape, cfy_shape]
                flowarrays = flowpacked.unpack(flowtensors)
                restensors = [compres, rex_shape, rey_shape]
                resarrays = respacked.unpack(restensors)

                fd = dict(zip(flowtensors, flowarrays))
                fd.update(dict(zip(restensors, resarrays)))
                fd.update(dict({testtfprvs: tenFirst}))
                tenFirst = sess.run(recimage, feed_dict=fd)
                sess.run(write_png(args.output + str(i * args.frequency + batch) + ".png", tenFirst))