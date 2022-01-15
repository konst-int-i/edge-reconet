import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa


class InstanceNorm(layers.Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        mean = tf.math.reduce_mean(x, axis=(1, 2), keepdims=True)
        recip_stdev = tf.math.rsqrt(
            tf.math.reduce_sum(
                tf.math.square(tf.math.subtract(x, mean)), axis=(1, 2), keepdims=True
            )
            / (216 * 512)
        )
        normed = tf.multiply(tf.math.subtract(x, mean), recip_stdev)
        return normed


class Normalization(layers.Layer):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = tf.reshape(mean, (1, 1, -1))
        self.std = tf.reshape(std, (1, 1, -1))

    def __call__(self, img):
        return (img - self.mean) / self.std


class ReconetNorm(layers.Layer):
    def __init__(self):
        super(ReconetNorm, self).__init__()

    def __call__(self, img):
        return (img * 2) - 1


class ReconetUnnorm(layers.Layer):
    def __init__(self):
        super(ReconetUnnorm, self).__init__()

    def __call__(self, img):
        return (img + 1) / 2


class ConvolutionalLayer(layers.Layer):
    def __init__(self, out_channels, kernel_size, stride, bias=True):
        super(ConvolutionalLayer, self).__init__()
        self.conv = layers.Conv2D(
            out_channels, kernel_size, strides=stride, use_bias=bias, padding="same"
        )

    def __call__(self, x):
        x = self.conv(x)
        return x


class ConvInstReLU(ConvolutionalLayer):
    def __init__(self, out_channels, kernel_size, stride):
        super(ConvInstReLU, self).__init__(out_channels, kernel_size, stride)
        self.inst = InstanceNorm()
        # self.inst = tfa.layers.InstanceNormalization()
        self.relu = activations.relu

    def __call__(self, x):
        x = super(ConvInstReLU, self).__call__(x)
        x = self.inst(x)
        x = self.relu(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, stride, padding="same")
        self.inst = InstanceNorm()
        # self.inst = tfa.layers.InstanceNormalization()

        self.relu = activations.relu

    def __call__(self, x):
        res = x
        x = self.relu(self.inst(self.conv(x)))
        x = self.inst(self.conv(x))
        x = res + x
        return x


class ReCoNet(tf.keras.Model):
    def __init__(self):
        super(ReCoNet, self).__init__()
        self.conv_inst_relu1 = ConvInstReLU(32, 9, 1)
        self.conv_inst_relu2 = ConvInstReLU(64, 3, 2)
        self.conv_inst_relu3 = ConvInstReLU(128, 3, 2)

        self.residual_block = ResBlock(128)

        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        # TODO - align_corners keyword of PT equivalent

        self.conv_inst_relu_dev1 = ConvInstReLU(64, 3, 1)
        self.conv_inst_relu_dev2 = ConvInstReLU(32, 3, 1)
        self.activation_conv = ConvolutionalLayer(3, 9, 1)
        self.tanh = activations.tanh

    def call(self, x):
        x = self.conv_inst_relu1(x)
        x = self.conv_inst_relu2(x)
        x = self.conv_inst_relu3(x)

        for _ in range(5):
            x = self.residual_block(x)

        feat_map = x

        x = self.upsample(x)
        x = self.conv_inst_relu_dev1(x)
        x = self.upsample(x)
        x = self.conv_inst_relu_dev2(x)
        x = self.activation_conv(x)
        image_output = self.tanh(x)

        return feat_map, image_output


if __name__ == "__main__":
    pass
