import tensorflow as tf
from PIL import Image
import numpy as np
from skimage.transform import resize
from timeit import default_timer as timer


def make_tflite_model(filepath):
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()

    return interpreter

def time_func(func, *args, **kwargs):
    before = timer()
    result = func(*args, **kwargs)
    after = timer()
    print(f"{func.__name__} took {after - before} seconds to run")
    return result

def tflite_infer(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    print(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    time_func(interpreter.invoke)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def get_image(path):
    im = Image.open(path)
    return im

def preprocess(im):
    im = np.array(im)
    im = resize(im, (216, 512))
    im = (im * 255).astype(np.uint8)
    im = np.expand_dims(im, axis=0)
    print(im.shape)
    # im = np.transpose(im)
    return im

def postprocess(im):
    return im

def write_image(im, outpath='test.png'):
    print(im)
    Image.fromarray(im.squeeze(0)).save(outpath)

def style_image_tflite(image, interpreter):
    image = preprocess(image)
    styled = tflite_infer(interpreter, image)
    print(f"{styled.shape}")
    styled = postprocess(styled)
    return styled


if __name__=="__main__":
    interpreter = make_tflite_model("saved_models/instance_normalised.tflite")
    image = get_image("images/alley01.png")
    styled = style_image(image, interpreter)
    write_image(styled)
