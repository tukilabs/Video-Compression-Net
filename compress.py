import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from utils.network import VideoCompressor
import numpy as np
from PIL import Image
import pickle as pkl
from utils.basics import write_png
import math
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="checkpoints/alloptimizedvideocomp.pkl",
                        help="Saved model that you want to compress with")

    parser.add_argument("--input", "-i", default="vimeo_septuplet/sequences/00001/0001/",
                        help="Directory where uncompressed frames lie and what you want to compress")

    parser.add_argument("--output", "-o", default="encoded/",
                        help="Directory where you want the compressed files to be saved"
                             "Warning: Output directory might have compressed files from"
                             "previous compression. If number of frames compressed previously"
                             "is greater than current number of frames then some of the compressed"
                             "files remains in the directory. During decompression these files"
                             "are also used for further reconstruction which may have bad output")

    parser.add_argument("--frequency", "-f", type=int, default=7,
                        help="Number of frames after which another image is passed to decoder"
                             "Should use same frequency during reconstruction")

    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()
    w, h, _ = np.array(Image.open(args.input + "im1.png")).shape
    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testsecond_frame")
    #
    num_pixels = w * h
    _, _, _ = testnet(testtfprvs, testtfnext)
    compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape, clipped_recon_image = testnet.compress(testtfprvs,
                                                                                                          testtfnext)
    flowtensors = [compflow, cfx_shape, cfy_shape]
    restensors = [compres, rex_shape, rey_shape]

    testinit = tf.global_variables_initializer()

    num_frames = 0
    for item in os.listdir(args.input):
        num_frames += 1

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        batch_range = args.frequency + 1
        for i in range(math.ceil(num_frames / args.frequency)):
            tenFirst = np.array(Image.open(args.input + 'im' + str(i * args.frequency + 1) + '.png')).astype(
                np.float32) * (1.0 / 255.0)
            tenFirst = np.expand_dims(tenFirst, axis=0)
            sess.run(write_png(args.output + str(i * args.frequency + 1) + ".png", tenFirst))

            if i == math.ceil(num_frames / args.frequency) - 1 and num_frames % args.frequency != 0:
                batch_range = num_frames % args.frequency + 1

            for batch in range(2, batch_range):
                tenSecond = np.array(Image.open(args.input + '/im' + str(i * args.frequency + batch) + '.png')).astype(
                    np.float32) * (1.0 / 255.0)
                tenSecond = np.expand_dims(tenSecond, axis=0)

                array_of, array_res, rec = sess.run([flowtensors, restensors, clipped_recon_image],
                                                    feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
                tenFirst = rec

                flowpacked = tfc.PackedTensors()
                flowpacked.pack(flowtensors, array_of)
                with open(args.output + "of" + str(i * args.frequency + batch - 1) + ".vcn", "wb") as f:
                    f.write(flowpacked.string)

                respacked = tfc.PackedTensors()
                respacked.pack(restensors, array_res)
                with open(args.output + "res" + str(i * args.frequency + batch - 1) + ".vcn", "wb") as f:
                    f.write(respacked.string)

                print("Actual_bpp = {:.8f}".format(((len(flowpacked.string) + len(respacked.string)) * 8 / num_pixels)))