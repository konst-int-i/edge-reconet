import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa


def resblock(x, kernelsize, filters):
    fx = layers.Conv2D(filters, kernelsize, activation="relu", padding="same")(x)
    fx = tfa.layers.InstanceNormalization()(fx)
    fx = layers.Conv2D(filters, kernelsize, padding="same")(fx)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    out = layers.Add()([x, fx])
    return out


def conv_instnorm_relu(x, filters, kernel_size, padding="same", strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = activations.relu(x)
    return x


def instance_normalisation(x):
    mean = tf.math.reduce_mean(x, axis=(1, 2))
    recip_stdev = tf.math.rsqrt(
        tf.math.reduce_sum(tf.math.squared_difference(x, mean), axis=(1, 2))
    )
    normed = tf.multiply(tf.math.subtract(x, mean), recip_stdev)
    return normed


def build_reconet() -> tf.keras.Model:
    # encoder_input = inputs['encoder_input']
    encoder_input = layers.Input(shape=(512, 216, 3), name="original_img")
    x = conv_instnorm_relu(encoder_input, 48, 9)
    x = conv_instnorm_relu(x, 96, 3, strides=2)
    x = conv_instnorm_relu(x, 192, 3, strides=2)

    x = resblock(x, 3, 192)
    x = resblock(x, 3, 192)
    x = resblock(x, 3, 192)
    x = resblock(x, 3, 192)
    x = resblock(x, 3, 192)

    features = x

    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = conv_instnorm_relu(x, 96, 3, padding="same")
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = conv_instnorm_relu(x, 48, 3, padding="same")
    out = layers.Conv2D(3, kernel_size=9, padding="same", activation="tanh")(x)
    return tf.keras.Model(inputs=encoder_input, outputs=[features, out])
