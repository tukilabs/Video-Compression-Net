import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from utils.network import VideoCompressor
import numpy as np
from PIL import Image
import pickle as pkl

if __name__ == "__main__":
    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, 256, 448, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, 256, 448, 3], name="testsecond_frame")
    #
    num_pixels = 256 * 448
    _, _, _ = testnet(testtfprvs, testtfnext)
    compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape = testnet.compress(testtfprvs, testtfnext)
    flowtensors = [compflow, cfx_shape, cfy_shape]
    restensors = [compres, rex_shape, rey_shape]

    flowpacked = tfc.PackedTensors()
    respacked = tfc.PackedTensors()

    testinit = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(testinit)
        with open("checkpoints/tfvideocomp.pkl", "rb") as f:
            testnet.set_weights(pkl.load(f))

        direc = "vimeo_septuplet/sequences/"
        outdir = "encoded/"
        subdir = direc + str(1).zfill(5) + '/'
        video_batch = 0
        tenFirst = subdir + str(4 * video_batch + 1).zfill(4) + '/im' + str(1) + '.png'
        tenFirst = np.array(Image.open(tenFirst)).astype(np.float32) * (1.0 / 255.0)
        tenFirst = np.expand_dims(tenFirst, axis=0)

        for batch in range(2, 8):
            tenSecond = subdir + str(4 * video_batch + 1).zfill(4) + '/im' + str(batch) + '.png'
            tenSecond = np.array(Image.open(tenSecond)).astype(np.float32) * (1.0 / 255.0)
            tenSecond = np.expand_dims(tenSecond, axis=0)

            array_of, array_res = sess.run([flowtensors, restensors],
                                           feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
            tenFirst = tenSecond
            #
            flowpacked.pack(flowtensors, array_of)
            with open(outdir + "of" + str(batch - 1) + ".tfci", "wb") as f:
                f.write(flowpacked.string)
            #
            respacked.pack(restensors, array_res)
            with open(outdir + "res" + str(batch - 1) + ".tfci", "wb") as f:
                f.write(respacked.string)

            print("Actual bpp = {:.8f}".format((len(flowpacked.string) + len(respacked.string)) * 8/num_pixels))