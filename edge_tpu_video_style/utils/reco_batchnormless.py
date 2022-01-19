
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa


def resblock(x, kernelsize, filters):
    fx = x
    x = layers.Conv2D(filters, kernelsize, activation="relu", padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernelsize, padding="same")(x)
    out = layers.Add()([x, fx])
    return out


def conv_layer(x, filters, kernel_size, padding="same", strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = activations.relu(x)
    return x



def build_reconet() -> tf.keras.Model:
    # encoder_input = inputs['encoder_input']
    encoder_input = layers.Input(shape=(216, 512, 3), name="original_img")
    x = conv_layer(encoder_input, 32, 9)
    x = conv_layer(x, 64, 3, strides=2)
    x = conv_layer(x, 128, 3, strides=2)

    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)
    x = resblock(x, 3, 128)

    features = x

    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = conv_layer(x, 64, 3, padding="same")
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = conv_layer(x, 32, 3, padding="same")
    out = layers.Conv2D(3, kernel_size=9, padding="same", activation="tanh")(x)
    return tf.keras.Model(inputs=encoder_input, outputs=[features, out])
