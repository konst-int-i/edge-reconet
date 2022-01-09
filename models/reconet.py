
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa
import os

import numpy as np
from timeit import default_timer as timer


def resblock(x, kernelsize, filters):
    fx = layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
    # fx = instance_normalisation(fx)
    fx = layers.Conv2D(filters, kernelsize, padding='same')(fx)
    out = layers.Add()([x,fx])
    # out = instance_normalisation(out)
    out = layers.ReLU()(out)
    return out


def conv_instnorm_relu(x, filters, kernel_size, padding='same', strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    # x = instance_normalisation(x)
    # x = tf.keras.layers.BatchNormalization(axis=[1,2])(x) 
    x = activations.relu(x)
    return x
    

def instance_normalisation(x):
    mean = tf.math.reduce_mean(x), #axis=(1, 2))
    recip_stdev = tf.math.rsqrt(tf.math.reduce_sum(tf.math.squared_difference(x, mean)))#, axis=(1, 2)))
    normed = tf.multiply(tf.math.subtract(x, mean), recip_stdev)
    return tf.math.subtract(x, mean)


# def instance_normalisation(x):
#     return tf.keras.layers.BatchNormalization(axis=[1,2])(x)


def build_model() -> tf.keras.Model:
    encoder_input = keras.Input(shape=(32, 32, 3), name="original_img")
    x = conv_instnorm_relu(encoder_input, 32, 9)
    x = layers.Conv2D(64, 3, padding='same', strides=(1, 1), activation="relu")(x)
    x = conv_instnorm_relu(x, 32, 3)
    
    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    # x = resblock(x, 3, 32)
    # x = resblock(x, 3, 32)
    # x = resblock(x, 3, 32)

    features = x

    x = conv_instnorm_relu(x, 32, 3, padding='same')
    # x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 9, padding='same', activation='tanh')(x)
    output = layers.Conv2D(3, 9, padding='same')(x)

    model = tf.keras.Model(inputs=encoder_input, outputs=output)

    return model


def quantise_model(model, model_name):
    saved_model_dir = 'saved_models/' + model_name
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8, tf.uint8]
    converter.inference_input_type = tf.uint8  
    converter.inference_output_type = tf.uint8  
    # converter.experimental_new_quantizer = True
    tflite_quant_model = converter.convert()

    return tflite_quant_model


def representative_dataset():
    for _ in range(100):
      data = tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1)
      yield [data]


def save_model(quantised, name='style_transfer.tflite'):
    with open('saved_models/name', 'wb') as f:
        f.write(quantised)


if __name__=="__main__":
    model = build_model()
    model.compile(loss='mse', optimizer='Adam')
    inp = tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1)
    print("Running model")
    before = timer()
    model.fit(inp, inp, epochs=1)
    res = model.predict(inp, batch_size=1)
    print("Finished running model")
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    model.save('saved_models/test_model')
    end = timer()
    print(f"Finished inference, {res.shape=}, time={end-before}, per sample={(end-before)/1}")

    print("quantising")
    quantised = quantise_model(model, 'test_model')

    save_model(quantised)
