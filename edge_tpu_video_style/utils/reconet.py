import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa


def resblock(x, kernelsize, filters):
    fx = x
    x = layers.Conv2D(filters, kernelsize, activation="relu", padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernelsize, padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    out = layers.Add()([x, fx])
    return out


def conv_instnorm_relu(x, filters, kernel_size, padding="same", strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = activations.relu(x)
    return x


def build_reconet() -> tf.keras.Model:
    encoder_input = layers.Input(shape=(None, None, 3), name="original_img")
    x = conv_instnorm_relu(encoder_input, 32, 9)
    x = conv_instnorm_relu(x, 64, 3, strides=2)
    x = conv_instnorm_relu(x, 128, 3, strides=2)

    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)

    features = x

    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = conv_instnorm_relu(x, 64, 3, padding="same")
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = conv_instnorm_relu(x, 32, 3, padding="same")
    out = layers.Conv2D(3, kernel_size=9, padding="same", activation="tanh")(x)
    return tf.keras.Model(inputs=encoder_input, outputs=[features, out])
