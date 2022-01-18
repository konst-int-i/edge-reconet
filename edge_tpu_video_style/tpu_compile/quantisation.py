import tensorflow as tf


def quantise_model(model_name):
    saved_model_dir = "saved_models/" + model_name
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
        data = tf.random.uniform((1, 512, 216, 3), minval=0, maxval=1)
        yield [data]
