import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from utils.network import VideoCompressor
import numpy as np
from PIL import Image
import pickle as pkl
from utils.basics import write_png


if __name__ == "__main__":
    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, 256, 448, 3], name="testfirst_frame")

    compflow = tf.placeholder(tf.string, [1], name="compressed_of_string")
    cfx_shape = tf.placeholder(tf.int32, [2], name="compressed_of_lengthx")
    cfy_shape = tf.placeholder(tf.int32, [2], name="compressed_of_lengthy")

    compres = tf.placeholder(tf.string, [1], name="compressed_residue_string")
    rex_shape = tf.placeholder(tf.int32, [2], name="compressed_residue_lengthx")
    rey_shape = tf.placeholder(tf.int32, [2], name="compressed_residue_lengthy")

    #call() should be called to call build()
    _, _, _ = testnet(testtfprvs, testtfprvs)

    # recres = testnet.rescomp.decompress(compres,rex_shape, rey_shape)
    recimage = testnet.decompress(testtfprvs, compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape)

    testinit = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(testinit)
        with open("checkpoints/tfvideocomp.pkl", "rb") as f:
            testnet.set_weights(pkl.load(f))

        outdir = "reconstructed/"
        direc = "vimeo_septuplet/sequences/"
        subdir = direc + str(1).zfill(5) + '/'
        video_batch = 0
        tenFirst = subdir + str(4 * video_batch + 1).zfill(4) + '/im' + str(1) + '.png'
        tenFirst = np.array(Image.open(tenFirst)).astype(np.float32) * (1.0 / 255.0)
        tenFirst = np.expand_dims(tenFirst, axis=0)
        sess.run(write_png(outdir + str(1) + ".png", tenFirst))

        for batch in range(2, 8):
            tenSecond = subdir + str(4 * video_batch + 1).zfill(4) + '/im' + str(batch) + '.png'
            tenSecond = np.array(Image.open(tenSecond)).astype(np.float32) * (1.0 / 255.0)
            tenSecond = np.expand_dims(tenSecond, axis=0)

            with open("encoded/" + "of" + str(batch - 1) + ".tfci", "rb") as f:
                flowpacked = tfc.PackedTensors(f.read())
            with open("encoded/" + "res" + str(batch - 1) + ".tfci", "rb") as f:
                respacked = tfc.PackedTensors(f.read())
                
            flowtensors = [compflow, cfx_shape, cfy_shape]
            flowarrays = flowpacked.unpack(flowtensors)
            restensors = [compres, rex_shape, rey_shape]
            resarrays = respacked.unpack(restensors)

            fd = dict(zip(flowtensors, flowarrays))
            fd.update(dict(zip(restensors, resarrays)))
            fd.update(dict({testtfprvs: tenFirst}))
            tenFirst = sess.run(recimage, feed_dict=fd)
            sess.run(write_png(outdir + str(batch) + ".png", tenFirst))

