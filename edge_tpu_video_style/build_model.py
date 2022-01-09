import os
from models.reconet import build_reconet
from models.utils import save_model
from postprocessing.quantisation import quantise_model
from edge_tpu_video_style.utils.parser import parser
from preprocessing.dataset import MPIDataSet
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.data import Dataset

import tensorflow as tf

from timeit import default_timer as timer


def build_toy_model():
    model = build_reconet()
    model.compile(loss="mse", optimizer="Adam")
    inp = tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1)
    print("Running model")
    before = timer()
    model.fit(inp, inp, epochs=1)
    res = model.predict(inp, batch_size=1)
    print("Finished running model")
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    model.save("saved_models/test_model")
    end = timer()
    print(
        f"Finished inference, {res.shape=}, time={end-before}, per sample={(end-before)/1}"
    )

    print("quantising")
    quantised = quantise_model(model, "test_model")

    save_model(quantised)


def train():
    pass


if __name__ == "__main__":
    # build_toy_model()
    args = parser.parse_args()

    # read in style image (weight, height)
    # style_dims = {"weight": 512, "height": 216}
    style_img = Image.open(args.style_name)
    style_img = style_img.resize((args.width, args.height), Image.BILINEAR)
    style_img = tf.convert_to_tensor(style_img)
    style_img = tf.transpose(style_img, (2, 0, 1))
    print(style_img)

    train_data = MPIDataSet(Path(args.path).joinpath("training"), args)
    test_data = MPIDataSet(Path(args.path).joinpath("test"), args)

    signature = (tf.uint8, tf.uint8, tf.uint8, tf.float32)
    # Convert to dataset
    train_dataset = Dataset.from_generator(
        lambda t: train_data, output_types=signature, name="train"
    )
    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer).batch(
        args.batch_size
    )
    test_dataset = Dataset.from_generator(
        lambda t: test_data, output_types=signature, name="test"
    )
    # test_data = Dataset.from_tensor_slices(test_data)

    print("train len:", len(train_dataset))
    print("test len:", len(test_dataset))
