import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

from timeit import default_timer as timer

from edge_tpu_video_style.models.utils import save_model
from edge_tpu_video_style.postprocessing.quantisation import quantise_model


def resblock(x, kernelsize, filters):
    fx = layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
    fx = instance_normalisation(fx)
    fx = layers.Conv2D(filters, kernelsize, padding='same')(fx)
    x = instance_normalisation(x)
    x = layers.ReLU()(x)
    out = layers.Add()([x,fx])
    return out


def conv_instnorm_relu(x, filters, kernel_size, padding="same", strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = instance_normalisation(x)
    x = activations.relu(x)
    return x


def instance_normalisation(x):
    mean = tf.math.reduce_mean(x, axis=(1, 2))
    recip_stdev = tf.math.rsqrt(tf.math.reduce_sum(tf.math.squared_difference(x, mean), axis=(1, 2)))
    normed = tf.multiply(tf.math.subtract(x, mean), recip_stdev)
    return normed


def build_reconet() -> tf.keras.Model:
    encoder_input = keras.Input(shape=(32, 32, 3), name="original_img")
    x = conv_instnorm_relu(encoder_input, 32, 9)
    x = layers.Conv2D(64, 3, padding="same", strides=(1, 1), activation="relu")(x)
    x = conv_instnorm_relu(x, 32, 3)

    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)

    features = x

    x = conv_instnorm_relu(x, 32, 3, padding='same')
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 9, padding='same', activation='tanh')(x)
    output = layers.Conv2D(3, 9, padding='same')(x)

    model = tf.keras.Model(inputs=encoder_input, outputs=output)

    return model
