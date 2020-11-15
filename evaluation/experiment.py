import tensorflow.compat.v1 as tf
from utils import VideoCompressor
import numpy as np
from PIL import Image
import pickle as pkl
import argparse
import os
import math
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
    parser.add_argument("--gop", "-g", type=int, default="10",
                        help="GOP size. \n"
                             "Default=10")

    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()
    if args.input[-1] is not '/':
        args.input += '/'

    w, h, _ = np.array(Image.open(os.path.join(args.input , "im1.png"))).shape

    if w % 16 != 0 or h % 16 != 0:
        raise ValueError('Height and Width must be mutiples of 16.')

    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testsecond_frame")

    recon_image, _, estimated_bpp = testnet(testtfprvs, testtfnext)
    msssim = tf.squeeze(tf.image.ssim_multiscale(recon_image, testtfnext, 1))
    orig = tf.round(tf.convert_to_tensor(tf.reshape(testtfnext, [w, h, 3])) * 255)
    rec = tf.round(tf.convert_to_tensor(tf.reshape(recon_image, [w, h, 3])) * 255)

    mse = tf.reduce_mean(tf.math.squared_difference(orig, rec))
    psnr = tf.squeeze(tf.image.psnr(rec, orig, 255))

    testinit = tf.global_variables_initializer()


    num_frames = len(os.listdir(args.input))

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        count = 0
        totmsssim = 0
        totpsnr = 0
        totbpp = 0

        batch_range = args.gop + 1
        for i in range(math.ceil(num_frames / args.gop)):
            tenFirst = np.array(Image.open(os.join.path(args.input , 'im' + str(i * args.gop + 1) + '.png'))).astype(np.float32) * (1.0 / 255.0)
            tenFirst = np.expand_dims(tenFirst, axis=0)

            if i == math.ceil(num_frames / args.gop) - 1 and num_frames % args.gop != 0:
                batch_range = num_frames % args.gop + 1

            for batch in range(2, batch_range):
                tenSecond = np.array(Image.open(os.path.join(args.input , 'im' + str(batch) + '.png'))).astype(np.float32) * (1.0 / 255.0)
                tenSecond = np.expand_dims(tenSecond, axis=0)

                reconimage, recloss, ps, msim, rate = sess.run([recon_image, mse, psnr, msssim, estimated_bpp],
                                                               feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
                tenFirst = reconimage

                totbpp += rate
                totmsssim += msim
                totpsnr += ps
                count += 1

        print("Average")
        print("psnr = {:.8f},  bpp = {:.8f}, ms-ssim ={:.8f}".format(totpsnr/count, totbpp/count, totmsssim/count))
