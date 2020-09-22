import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import numpy as np
from .basics import warp


class SpyNetwork(tf.keras.layers.Layer):
    """
    Spatial Pyramid Network to compute optical flow
    Original Paper: https://arxiv.org/abs/1611.00850
    """
    def __init__(self, *args, **kwargs):
        """
        Initializer.
        Arguments:
          **kwargs: Other keyword arguments passed to superclass tf.keras.layers.Layer.
        """
        super(SpyNetwork, self).__init__(*args, **kwargs)

        class Preprocess(tf.keras.layers.Layer):
            """
            Normalizes BGR components.
            """
            def __init__(self, *arguments, **keywordargs):
                """Initializer.
                Arguments:
                  **kwargs: Other keyword arguments passed to superclass tf.keras.layers.Layer.
                """
                super(Preprocess, self).__init__(*arguments, **keywordargs)

            def call(self, teninput, **keywordargs):
                """
                Arguments:
                  teninput: A BGR image in range [0,1]

                Returns:
                    Normalized Image.
                    The means [0.406, 0.456, 0.485] and std deviation [0.225 0.224 0.229]
                    of respective colors are obtained from ImageNet dataset. This is true
                    for all natural images.
                """
                tenblue = (teninput[:, :, :, 0:1] - 0.406) / 0.225
                tengreen = (teninput[:, :, :, 1:2] - 0.456) / 0.224
                tenred = (teninput[:, :, :, 2:3] - 0.485) / 0.229
                return tf.concat([tenblue, tengreen, tenred], 3)

        class Basic(tf.keras.layers.Layer):
            """
            A SPYnet architecture
            TODO: Fiddle around the number of channels, kernel size, stride and padding if it works better
            """
            def __init__(self, *arguments, **keywordargs):
                """Initializer.
                Arguments:
                   **kwargs: Other keyword arguments passed to superclass tf.keras.layers.Layer
                """
                super(Basic, self).__init__(*arguments, **keywordargs)

            def build(self, input_shape):
                self.netBasic = tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding="same",
                                               kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding="same",
                                               kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding="same",
                                               kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(filters=16, kernel_size=7, strides=1, padding="same",
                                               kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(filters=2, kernel_size=7, strides=1, padding="same",
                                               kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                    ]
                )
                super(Basic, self).build(input_shape)

            def call(self, teninput, **keywordargs):
                """
                Arguments:
                    teninput: Input with 8 channels 3-3 each for two consecutive image frame and 2 for
                    the flow that is initially initialized to 0

                Returns:
                    Neural Computed flow with 2 channels
                """
                return self.netBasic(teninput)

        self.netPreprocess = Preprocess()
        self.netBasic = [Basic() for _ in range(5)]

    def build(self, input_shape):
        super(SpyNetwork, self).build(input_shape)

    def call(self, tenfirst, tensecond):
        tenfirst = [self.netPreprocess(tenfirst)]
        tensecond = [self.netPreprocess(tensecond)]

        if max(tenfirst[0].shape[1], tenfirst[0].shape[2]) > 512:
            num_layer = 4
        else:
            num_layer = 5

        for intLevel in range(num_layer):
            if tenfirst[0].shape[1] > 32 or tenfirst[0].shape[2] > 32:
                tenfirst.insert(0, tf.keras.layers.AveragePooling2D(pool_size=2)(tenfirst[0]))
                tensecond.insert(0, tf.keras.layers.AveragePooling2D(pool_size=2)(tensecond[0]))

        tenflow = tf.zeros([tenfirst[0].shape[0], tenfirst[0].shape[1] // 2, tenfirst[0].shape[2] // 2, 2])

        for intLevel in range(min(len(tenfirst), 5)):
            tenupsampled = tf.image.resize_bilinear(tenflow, [tenflow.shape[1] * 2, tenflow.shape[2] * 2]) * 2.0

            if tenupsampled.shape[1] != tenfirst[intLevel].shape[1] or tenupsampled.shape[2] != tenfirst[intLevel].shape[2]:
                tenupsampled = tf.pad(tenupsampled, [[0, 0], [tenfirst[intLevel].shape[1] - tenupsampled.shape[1], 0],
                                                     [tenfirst[intLevel].shape[2] - tenupsampled.shape[2], 0],[0, 0]],
                                      "SYMMETRIC")
            tenflow = self.netBasic[intLevel](
                tf.concat(
                    [tenfirst[intLevel], warp(tensecond[intLevel], -tenupsampled), tenupsampled], 3)
            ) + tenupsampled

        return tenflow


class AnalysisTransform(tf.keras.layers.Layer):
    """
    Encodes the optical flow and residue to their respective latent representations
    Based on : https://arxiv.org/abs/1802.01436 by Balle et al.
    """
    def __init__(self, num_filters, *args, **kwargs):
        """
        Initializer.
        Arguments:
            num_filters = number of convolutional filters in each intermediate layers
            *args, **kwargs = Other arguments and keyword arguments passed to superclass
        """
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),

            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),

            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_2")),

            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_3", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor, **kwargs):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """
    Reconstructs the optical flow and residue from their respective latent representations
    Based on : https://arxiv.org/abs/1802.01436 by Balle et al.
    """
    def __init__(self, num_channels, num_filters, *args, **kwargs):
        """
        Initializer.
        Arguments:
            num_channels = number of output channels. 2, 3 for optical flow and residue respectively
            num_filters = number of convolutional filters in each intermediate layers
            *args, **kwargs = Other arguments and keyword arguments passed to superclass
        """
        self.num_filters = num_filters
        self.num_channels = num_channels
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_0", corr=True, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True)),

            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_1", corr=True, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_2", corr=True, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True)),
            tfc.SignalConv2D(
                self.num_channels, (3, 3), name="layer_3", corr=True, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor, **kwargs):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class ImageCompressor(tf.keras.layers.Layer):
    """
    Optical Flow and/or Residue compression
    Uses EntropyBottleneck class from https://github.com/tensorflow/compression/
    for bitrate estimation and Entrppy coding
    """

    def __init__(self, num_channels, num_filters, training=True, *args, **kwargs):
        """
        num_channels = number of output channels. 2, 3 for optical flow and residue respectively
        num_filters = number of convolutional filters in each intermediate layers
        training = True for training and False for evaluation
        *args, **kwargs = Other arguments and keyword arguments passed to superclass
        """
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.training = training
        super(ImageCompressor, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.analysis_transform = AnalysisTransform(num_filters=self.num_filters)
        self.entropy_bottleneck = tfc.EntropyBottleneck()
        self.synthesis_transform = SynthesisTransform(num_channels=self.num_channels, num_filters=self.num_filters)
        super(ImageCompressor, self).build(input_shape)

    def call(self, tensor, **kwargs):
        y = self.analysis_transform(tensor)
        y_tilde, likelihoods = self.entropy_bottleneck(y, training=self.training)
        x_tilde = self.synthesis_transform(y_tilde)
        total_bits = tf.reduce_sum(tf.math.log(likelihoods)) / (-np.log(2))
        return x_tilde, total_bits

    def compress(self, tensor):
        y = self.analysis_transform(tensor)
        string = self.entropy_bottleneck.compress(y)

        y_hat, likelihoods = self.entropy_bottleneck(y, training=False)

        eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2))
        return string, tf.shape(tensor)[1:-1], tf.shape(y)[1:-1], eval_bpp
        # return string, tf.shape(tensor)[1:-1], tf.shape(y)[1:-1]


    def decompress(self, string, x_shape, y_shape):
        y_shape = tf.concat([y_shape, [self.num_filters]], axis=0)
        y_hat = self.entropy_bottleneck.decompress(
            string, y_shape, channels=self.num_filters)
        x_hat = self.synthesis_transform(y_hat)
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        return x_hat

class VideoCompressor(tf.keras.layers.Layer):
    """
    Computes the optical flow between consecutive frames, compress the thus obtained optical flow,
    decompress the compressed optical flow, warps previously reconstructed image with the decompressed
    optical flow to obtain motion compensated frame. Compress the mismatch in motion compensated frame
    and original frame and decompress it to obtain reconstructed residue. Sum reconstructed optical flow
    and residue to obtain reconstructed frame. To train the network, objective should be to minimize the
    weighted sum of distortion (mismatch in original frame and reconstructed frame) and bitrate of compressed
    optical flow and residue
    """
    def __init__(self, training=True, *args, **kwargs):
        """
        training = True for training and False for evaluation
        *args, **kwargs = Other arguments and keyword arguments passed to superclass
        """
        self.training = training
        super(VideoCompressor, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.ofnet = SpyNetwork()
        self.ofcomp = ImageCompressor(num_channels=2, num_filters=128, training=self.training)
        self.rescomp = ImageCompressor(num_channels=3, num_filters=128, training=self.training)
        super(VideoCompressor, self).build(input_shape)

    def call(self, prevreconstructed, tensecond):
        tenflow = self.ofnet(prevreconstructed, tensecond)
        reconflow = self.ofcomp(tenflow)
        motionCompensated = warp(prevreconstructed, reconflow[0])
        res = tensecond - motionCompensated
        reconres = self.rescomp(res)
        recon_image = motionCompensated + reconres[0]
        clipped_recon_image = tf.clip_by_value(recon_image, 0, 1)
        mse_loss = tf.reduce_mean(tf.math.squared_difference(recon_image, tensecond))
        # mse_loss = 1 - tf.math.reduce_mean(tf.image.ssim_multiscale(clipped_recon_image, tensecond, max_val=1))
        # comment the uncommented mse_loss and uncomment the commented mse_loss to use MS-SSIM as recontruction loss
        # rather than MSE loss
        total_bits_feature = reconres[1] + reconflow[1]
        batch_size, height, width, _ = prevreconstructed.shape
        bpp_feature = tf.divide(tf.cast(total_bits_feature, tf.float32),
                                tf.cast(batch_size * height * width, tf.float32))

        return clipped_recon_image, mse_loss, bpp_feature

    def compress(self, prevreconstructed, tensecond):
        tenflow = self.ofnet(prevreconstructed, tensecond)
        compflow, cfx_shape, cfy_shape, of_bpp = self.ofcomp.compress(tenflow)
        reconflow = self.ofcomp.decompress(compflow, cfx_shape, cfy_shape)
        motionCompensated = warp(prevreconstructed, reconflow)
        res = tensecond - motionCompensated
        compres, rex_shape, rey_shape, res_bpp = self.rescomp.compress((res))
        reconres = self.rescomp.decompress(compres, rex_shape, rey_shape)
        recon_image = motionCompensated + reconres
        clipped_recon_image = tf.clip_by_value(recon_image, 0, 1)
        return compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape, clipped_recon_image


    def decompress(self, prevreconstructed, compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape):
        reconflow = self.ofcomp.decompress(compflow, cfx_shape, cfy_shape)
        reconres = self.rescomp.decompress(compres, rex_shape, rey_shape)
        motionCompensated = warp(prevreconstructed, reconflow)
        recon_image = motionCompensated + reconres
        clipped_recon_image = tf.clip_by_value(recon_image, 0, 1)
        return clipped_recon_image