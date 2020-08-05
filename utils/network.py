import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import numpy as np
from .basics import warp


class SpyNetwork(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SpyNetwork, self).__init__(*args, **kwargs)

        class Preprocess(tf.keras.layers.Layer):
            def __init__(self, *arguments, **keywordargs):
                super(Preprocess, self).__init__(*arguments, **keywordargs)

            def call(self, teninput, **keywordargs):
                tenblue = (teninput[:, :, :, 0:1] - 0.406) / 0.225
                tengreen = (teninput[:, :, :, 1:2] - 0.456) / 0.224
                tenred = (teninput[:, :, :, 2:3] - 0.485) / 0.229
                return tf.concat([tenblue, tengreen, tenred], 3)

        class Basic(tf.keras.layers.Layer):
            def __init__(self, *arguments, **keywordargs):
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
                return self.netBasic(teninput)

        self.netPreprocess = Preprocess()
        self.netBasic = [Basic() for _ in range(5)]

    def build(self, input_shape):
        super(SpyNetwork, self).build(input_shape)

    def call(self, tenfirst, tensecond):
        tenfirst = [self.netPreprocess(tenfirst)]
        tensecond = [self.netPreprocess(tensecond)]

        for intLevel in range(5):
            if tenfirst[0].shape[1] > 32 or tenfirst[0].shape[2] > 32:
                tenfirst.insert(0, tf.keras.layers.AveragePooling2D(pool_size=2)(tenfirst[0]))
                tensecond.insert(0, tf.keras.layers.AveragePooling2D(pool_size=2)(tensecond[0]))

        tenflow = tf.zeros([tenfirst[0].shape[0], tenfirst[0].shape[1] // 2, tenfirst[0].shape[2] // 2, 2])

        for intLevel in range(len(tenfirst)):
            tenupsampled = tf.image.resize_bilinear(tenflow, [tenflow.shape[1] * 2, tenflow.shape[2] * 2]) * 2.0

            tenflow = self.netBasic[intLevel](
                tf.concat(
                    [tenfirst[intLevel], warp(tensecond[intLevel], -tenupsampled), tenupsampled], 3)
            ) + tenupsampled

        return tenflow


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
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
    """The synthesis transform."""

    def __init__(self, num_channels, num_filters, *args, **kwargs):
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
  """

    def __init__(self, num_channels, num_filters, training=True, *args, **kwargs):
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
        return string, tf.shape(tensor)[1:-1], tf.shape(y)[1:-1]

    def decompress(self, string, x_shape, y_shape):
        y_shape = tf.concat([y_shape, [self.num_filters]], axis=0)
        y_hat = self.entropy_bottleneck.decompress(
            string, y_shape, channels=self.num_filters)
        x_hat = self.synthesis_transform(y_hat)
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        return x_hat

class VideoCompressor(tf.keras.layers.Layer):
    """
    """
    def __init__(self, training=True, *args, **kwargs):
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
        total_bits_feature = reconres[1] + reconflow[1]
        batch_size, height, width, _ = prevreconstructed.shape
        bpp_feature = tf.divide(tf.cast(total_bits_feature, tf.float32),
                                tf.cast(batch_size * height * width, tf.float32))

        return clipped_recon_image, mse_loss, bpp_feature

    def compress(self, prevreconstructed, tensecond):
        tenflow = self.ofnet(prevreconstructed, tensecond)
        compflow, cfx_shape, cfy_shape = self.ofcomp.compress(tenflow)
        reconflow = self.ofcomp(tenflow)
        motionCompensated = warp(prevreconstructed, reconflow[0])
        res = tensecond - motionCompensated
        compres, rex_shape, rey_shape = self.rescomp.compress((res))
        return compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape

    def decompress(self, prevreconstructed, compflow, cfx_shape, cfy_shape, compres, rex_shape, rey_shape):
        reconflow = self.ofcomp.decompress(compflow, cfx_shape, cfy_shape)
        reconres = self.rescomp.decompress(compres, rex_shape, rey_shape)
        motionCompensated = warp(prevreconstructed, reconflow)
        recon_image = motionCompensated + reconres
        clipped_recon_image = tf.clip_by_value(recon_image, 0, 1)
        return clipped_recon_image