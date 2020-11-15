import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from utils import VideoCompressor, write_png
import numpy as np
from PIL import Image
import pickle as pkl
import math
import argparse
import os
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="checkpoints/videocompressor1024.pkl",
                        help="Saved model that you want to compress with\n"
                             "Default=`checkpoints/videocompressor1024.pkl`")

    parser.add_argument("--input", "-i", default="demo/input/",
                        help="Directory where uncompressed frames lie and what you want to compress\n"
                             "Default=`demo/input/`")

    parser.add_argument("--output", "-o", default="demo/compressed/",
                        help="Directory where you want the compressed files to be saved\n"
                             "Warning: Output directory might have compressed files from\n"
                             "previous compression. If number of frames compressed previously\n"
                             "is greater than current number of frames then some of the compressed\n"
                             "files remains in the directory. During decompression these files\n"
                             "are also used for further reconstruction which may have bad output\n"
                             "Default=`demo/compressed/`")

    parser.add_argument("--frequency", "-f", type=int, default=7,
                        help="Number of frames after which another image is passed to decoder\n"
                             "Should use same frequency during reconstruction. \n"
                             "Default=7")

    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    w, h, _ = np.array(Image.open(os.path.join(args.input , "im1.png"))).shape

    if w % 16 != 0 or h % 16 != 0:
        raise ValueError('Height and Width must be mutiples of 16.')

    if os.listdir(args.output):
        print("Warning: {} is not-empty. There might be an issue while decompression".format((args.output)))

    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testsecond_frame")
    #
    num_pixels = w * h
    testnet(testtfprvs, testtfnext)  #required to call call() to call build()
    compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape, clipped_recon_image = testnet.compress(testtfprvs,
                                                                                                          testtfnext)
    flowtensors = [compflow, cfx_shape, cfy_shape]
    restensors = [compres, rex_shape, rey_shape]

    testinit = tf.global_variables_initializer()

    num_frames = len(os.listdir(args.input))

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        batch_range = args.frequency + 1
        for i in range(math.ceil(num_frames / args.frequency)):
            tenFirst = np.array(Image.open(os.path.join(args.input , 'im' + str(i * args.frequency + 1) + '.png'))).astype(
                np.float32) * (1.0 / 255.0)
            tenFirst = np.expand_dims(tenFirst, axis=0)
            sess.run(write_png(os.path.join(args.output , str(i * args.frequency + 1) + '.png'), tenFirst))

            if i == math.ceil(num_frames / args.frequency) - 1 and num_frames % args.frequency != 0:
                batch_range = num_frames % args.frequency + 1

            for batch in range(2, batch_range):
                tenSecond = np.array(Image.open(os.path.join(args.input , 'im' + str(i * args.frequency + batch) + '.png'))).astype(
                    np.float32) * (1.0 / 255.0)
                tenSecond = np.expand_dims(tenSecond, axis=0)

                array_of, array_res, rec = sess.run([flowtensors, restensors, clipped_recon_image],
                                                    feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
                tenFirst = rec

                flowpacked = tfc.PackedTensors()
                flowpacked.pack(flowtensors, array_of)
                with open(os.path.join(args.output, "of" + str(i * args.frequency + batch - 1) + ".vcn"), "wb") as f:
                    f.write(flowpacked.string)

                respacked = tfc.PackedTensors()
                respacked.pack(restensors, array_res)
                with open(os.path.join(args.output , "res" + str(i * args.frequency + batch - 1) + ".vcn"), "wb") as f:
                    f.write(respacked.string)

                print("Actual_bpp = {:.8f}".format(((len(flowpacked.string) + len(respacked.string)) * 8 / num_pixels)))