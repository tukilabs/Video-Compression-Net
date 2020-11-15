import tensorflow.compat.v1 as tf
from utils import VideoCompressor, write_png
import numpy as np
from PIL import Image
import pickle as pkl
import argparse
import os
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="checkpoints/videocompressor1024.pkl",
                        help="Saved model that you want to test with\n"
                             "Default=`checkpoints/videocompressor1024.pkl`")
    parser.add_argument("--input", "-i", default="demo/input/",
                        help="Directory where uncompressed frames lie and what you want to compress\n"
                             "Default=`demo/input/`")
    parser.add_argument("--output", "-o", default="demo/reconstructed/",
                        help="Directory where you want the reconstructed frames to be saved. \n"
                             "Default=`demo/reconstructed/`")
    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    w, h, _ = np.array(Image.open(os.path.join(args.input , "im1.png"))).shape

    if w % 16 != 0 or h % 16 != 0:
        raise ValueError('Height and Width must be mutiples of 16.')

    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testsecond_frame")

    recon_image, _, estimated_bpp = testnet(testtfprvs, testtfnext)
    orig = tf.round(tf.convert_to_tensor(tf.reshape(testtfnext, [w, h, 3])) * 255)
    rec = tf.round(tf.convert_to_tensor(tf.reshape(recon_image, [w, h, 3])) * 255)

    mse = tf.reduce_mean(tf.math.squared_difference(orig, rec))
    psnr = tf.squeeze(tf.image.psnr(rec, orig, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(rec, orig, 255))

    testinit = tf.global_variables_initializer()

    num_frames = len(os.listdir(args.input))

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        tenFirst = np.array(Image.open(os.path.join(args.input , 'im' + str(1) + '.png'))).astype(np.float32) * (1.0 / 255.0)
        tenFirst = np.expand_dims(tenFirst, axis=0)
        sess.run(write_png(os.path.join(args.output , str(1) + ".png"), tenFirst))

        for batch in range(2, num_frames + 1):
            tenSecond = np.array(Image.open(os.path.join(args.input, 'im' + str(batch) + '.png'))).astype(np.float32) * (1.0 / 255.0)
            tenSecond = np.expand_dims(tenSecond, axis=0)

            reconimage, recloss, ps, ms, rate = sess.run([recon_image, mse, psnr, msssim, estimated_bpp],
                                                         feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
            tenFirst = reconimage

            print("recon loss = {:.8f}, psnr = {:.8f}, msssim = {:.8f}, bpp = {:.8f}".format(recloss, ps, ms, rate))
            sess.run(write_png(os.path.join(args.output, str(batch) + ".png"), reconimage))