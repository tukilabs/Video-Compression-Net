import tensorflow.compat.v1 as tf
from utils import VideoCompressor, write_png, warp
import numpy as np
from PIL import Image
import pickle as pkl
import cv2
import argparse
import os

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def flow_to_img(flow, normalize=True, info=None, flow_mag_max=None):
    """Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.
    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
        info: Text to superimpose on image (typically, the epe for the predicted flow)
        flow_mag_max: Max flow to map to 255
    Returns:
        img: viewable representation of the dense optical flow in RGB format
        flow_avg: optionally, also return average flow magnitude
    Ref:
        - OpenCV 3.0.0-dev documentation » OpenCV-Python Tutorials » Video Analysis »
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    flow = np.squeeze(flow, axis=0)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return np.expand_dims(img, axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="checkpoints/videocompressor8192.pkl",
                        help="Saved model that you want to analyse\n"
                             "Default=`checkpoints/videocompressor8192.pkl`")

    parser.add_argument("--input", "-i", default="demo/input/",
                        help="Directory where uncompressed frames lie and what you want to analyze\n"
                             "Default=`demo/input/`")

    parser.add_argument("--output", "-o", default="demo/visualization/",
                        help="Directory where you want the analyzed files to be saved\n"
                             "Default=`demo/visualization/`")

    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()
    if args.input[-1] is not '/':
        args.input += '/'

    if args.output[-1] is not '/':
        args.output += '/'

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    w, h, _ = np.array(Image.open(args.input + "im1.png")).shape

    if w % 16 != 0 or h % 16 != 0:
        raise ValueError('Height and Width must be mutiples of 16.')

    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testsecond_frame")
    #
    _, _, _ = testnet(testtfprvs, testtfnext)
    flow = testnet.ofnet(testtfprvs, testtfnext)
    reconflow, _ = testnet.ofcomp(flow)
    motionCompensated = warp(testtfprvs, reconflow)
    res = testtfnext - motionCompensated
    reconres, _ = testnet.rescomp(res)

    recon_image = motionCompensated + reconres

    testinit = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        tenFirst = np.array(Image.open(args.input + 'im1.png')).astype(np.float32) * (1.0 / 255.0)
        tenFirst = np.expand_dims(tenFirst, axis=0)
        tenSecond = np.array(Image.open(args.input + 'im2.png')).astype(np.float32) * (1.0 / 255.0)
        tenSecond = np.expand_dims(tenSecond, axis=0)

        realflow, realreconflow, realmotcom, realres, realreconres, realimage = sess.run([flow, reconflow, motionCompensated,
                                                                                             res, reconres, recon_image],
                                                                                            feed_dict={testtfprvs: tenFirst,
                                                                                                       testtfnext: tenSecond})

        realflow = flow_to_img(realflow)
        realreconflow = flow_to_img(realreconflow)
        sess.run(write_png(args.output + "flow.png", realflow))
        sess.run(write_png(args.output + "reconflow.png", realreconflow))
        # pkl.dump(np.squeeze(realflow, axis=0), open(args.output + "of.flo", "wb"))
        # pkl.dump(np.squeeze(realreconflow, axis=0), open(args.output + "reconof.flo", "wb"))
        sess.run(write_png(args.output + "first.png", tenFirst))
        sess.run(write_png(args.output + "second.png", tenSecond))
        sess.run(write_png(args.output + "motioncompensated.png", realmotcom))
        sess.run(write_png(args.output + "residue.png", realres))
        sess.run(write_png(args.output + "reconresidue.png", realreconres))
        sess.run(write_png(args.output + "reconstructed.png", realimage))
