import cv2
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
from utils.style_quantised import style_image_tflite, make_tflite_model
from utils.parser import parser

model = tf.keras.models.load_model("saved_models/reconet_tf_in.pb")


def run_inference(image):
    tf_im = model(tf.expand_dims(tf.Variable(image, dtype=tf.float32), axis=0))
    return tf_im[1]


inference_size = (400, 216)


def min_max_scale_tf(tensor):
    return (tensor - tf.reduce_min(tensor, keepdims=True)) / (
        tf.reduce_max(tensor, keepdims=True) - tf.reduce_min(tensor, keepdims=True)
    )


def preprocess_normal(inp):
    inp = inp / 255.0
    inp = inp / 2 - 1
    inp = np.expand_dims(inp, axis=0)
    return inp


def postprocess_normal(output):
    # output = tf.nn.sigmoid(output)
    output = output[1].numpy()
    # output = output_tensor(interpreter, 0)[0]
    output = (output - output.min()) / (output.max() - output.min())
    return output * 255


def run_inference(image, model):
    image = preprocess_normal(image)
    image = model(image)
    image = postprocess_normal(image)
    return image


def show_webcam(
    model,
    mirror=False,
    input_location="/dev/video0",
    resolution=(640, 480),
    output_size=(512, 216),
    output_location="display",
    mode="normal",
):

    cam = cv2.VideoCapture(input_location)
    if output_location != "display":
        out = cv2.VideoWriter(
            output_location, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 24, output_size
        )
    while True:
        ret_val, img = cam.read()
        if not ret_val:
            break

        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(img, resolution)
        inp = cv2_im_rgb

        if mode == "tflite":
            output = style_image_tflite(inp, model)
        elif mode == "tflite-no-inst-norm":
            output = style_image_tflite(inp, model)
        else:
            output = run_inference(inp, model)

        image = cv2.resize(output[0, :, :, :].astype(np.uint8), output_size)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if mirror:
            img = cv2.flip(img, 1)
        if output_location == "display":
            cv2.imshow("frame", image)
        else:
            out.write(image)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()


def main():
    args = parser.parse_args()
    model = tf.keras.models.load_model("saved_models/reconet_tf_in.pb")
    show_webcam(
        mirror=True,
        input_location=args.video_input,
        resolution=args.resolution,
        model=model,
        mode="normal",
        output_size=(1080, 760),
        output_location=args.video_output,
    )


if __name__ == "__main__":
    main()
