import tensorflow as tf
import math
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa


class Normalization(tf.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = tf.reshape(mean, (-1, 1, 1))
        self.std = tf.reshape(std, (-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


class ConvolutionalLayer(tf.Module):
    def __init__(self, out_channels, kernel_size, stride, bias=True):
        super(ConvolutionalLayer, self).__init__()
        padd = int(math.floor(kernel_size / 2))
        # self.reflect = ReflectionPadding2D(padd)
        self.conv = layers.Conv2D(
            out_channels, kernel_size, strides=stride, use_bias=bias, padding="same"
        )

    def __call__(self, x):
        x = self.conv(x)
        print(x.shape)
        return x


class ConvInstReLU(ConvolutionalLayer):
    def __init__(self, out_channels, kernel_size, stride):
        super(ConvInstReLU, self).__init__(out_channels, kernel_size, stride)
        self.inst = tfa.layers.InstanceNormalization()
        self.relu = activations.relu

    def __call__(self, x):
        # print(x)
        x = super(ConvInstReLU, self).__call__(x)
        x = self.inst(x)
        x = self.relu(x)
        return x


class ResBlock(tf.Module):
    def __init__(self, filters, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, stride, padding="same")
        self.instnorm = tfa.layers.InstanceNormalization()
        # self.second_conv = layers.Conv2D(filters, kernel_size, stride, padding="same")
        # self.se = tfa.InstanceNormalization(filters)
        self.relu = activations.relu

    # def forward(self, x):
    def __call__(self, x):
        res = x
        x = self.relu(self.instnorm(self.conv(x)))
        x = self.instnorm(self.conv(x))
        x = res + x
        return x


class ReCoNet(tf.Module):
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

    def __call__(self, x):
        x = self.conv_inst_relu1(x)
        x = self.conv_inst_relu2(x)
        x = self.conv_inst_relu3(x)

        for _ in range(5):
            x = self.residual_block(x)

        feat = x

        # for _ in range(2):
        x = self.upsample(x)
        x = self.conv_inst_relu_dev1(x)
        x = self.upsample(x)
        x = self.conv_inst_relu_dev2(x)
        x = self.activation_conv(x)
        x = self.tanh(x)

        return feat, x


if __name__ == "__main__":
    torch.manual_seed(0)
    input_pt = torch.randn(1, 3, 216, 512)
    input_np = input_pt.numpy()
    input = tf.convert_to_tensor(input_np)
    x = tf.transpose(input, (0, 3, 2, 1))
    # print(input.shape)
    # test convlayer
    model = ReCoNet()
    feat, x = model(x)
    # print(feat)
    # model = ConvolutionalLayer(32, 9, 1)
    # cir1 = ConvInstReLU(32, 9, 1)
    # cir2 = ConvInstReLU(64, 3, 2)
    # cir3 = ConvInstReLU(128, 3, 2)
    # res = ResBlock(128)
    # x = cir1(x)
    # x = cir2(x)
    # x = cir3(x)
    # x = res(x)

    print(tf.transpose(x, (0, 3, 2, 1)))
    # print(tf.transpose(feat, (0, 3, 2, 1)))
    # print(x)
