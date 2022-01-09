import argparse
import os
from models.reconet import build_reconet
from models.utils import save_model
from postprocessing.quantisation import quantise_model
from edge_tpu_video_style.utils.parser import parser
from models.layers import Normalization
from preprocessing.dataset import MPIDataSet
from PIL import Image
from pathlib import Path
from tqdm import tqdm
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


def train(
    args: argparse.Namespace,
    train_dataset: Dataset,
    test_dataset: Dataset,
    norm_mean: tf.Tensor,
    norm_std: tf.Tensor,
):

    normalization = Normalization(norm_mean, norm_std)

    style_img_list = [style_img]

    # for epoch in range(args.epochs):
    for epoch in range(1):  # TODO - change to actual epochs
        data_bar = tqdm(train_dataset)
        for id, sample in enumerate(data_bar):
            print(id)
            if id == 2:
                break
            # (img1, img2, mask, flow) = sample
            img1, img2, mask, flow = sample
            img1, img2 = img2, img1

            # feat1, output_img1 = style_model(img1)
            # feat2, output_img2 = style_model(img2)


if __name__ == "__main__":
    # build_toy_model()
    args = parser.parse_args()
    print(type(args))

    # TODO - initiate GPU training (to CUDA device)
    cnn_normalization_mean = tf.constant([0.485, 0.456, 0.406])
    cnn_normalization_std = tf.constant([0.229, 0.224, 0.225])

    # model_style = ReCoNet()

    # read in style image (weight, height)
    # style_dims = {"weight": 512, "height": 216}
    style_img = Image.open(args.style_name)
    style_img = style_img.resize((args.width, args.height), Image.BILINEAR)
    style_img = tf.convert_to_tensor(style_img)
    # style_img = tf.transpose(style_img, (2, 0, 1)) # no need to transpose since tf tensors are ordered differently
    print(style_img)

    train_data = MPIDataSet(Path(args.path).joinpath("training"), args)
    test_data = MPIDataSet(Path(args.path).joinpath("test"), args)

    # for idx, sample in enumerate(train_data):
    #     if idx == 2:
    #         break
    #     img1, img2, mask, flow = sample

    signature = (tf.uint8, tf.uint8, tf.uint8, tf.float32)
    # Convert to dataset
    train_dataset = Dataset.from_generator(
        train_data, output_types=signature, name="train"
    )

    # train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer).batch(
    #     args.batch_size
    # )
    train_dataset = train_dataset.batch(args.batch_size)
    test_dataset = Dataset.from_generator(
        test_data, output_types=signature, name="test"
    )

    # train model
    train(
        args=args,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        norm_mean=cnn_normalization_mean,
        norm_std=cnn_normalization_std,
    )

    # print("train len:", len(train_dataset))
    # print("test len:", len(test_dataset))
