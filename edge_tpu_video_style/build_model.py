import argparse
import os
from models.reconet import build_reconet
from models.utils import save_model, warp_back, calculate_luminance_mask
from postprocessing.quantisation import quantise_model
from edge_tpu_video_style.utils.parser import parser
from models.layers import ReCoNet
from models.layers import Normalization
from preprocessing.dataset import MPIDataSet
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras import layers, losses, optimizers
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.data import Dataset
import tensorflow_addons as tfa

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
    style_model: tf.Module,
    loss_model: tf.keras.Model,
    mse_loss,
    sum_mse_loss,
    optimizer,
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
            img1, img2, mask, flow = sample
            img1, img2 = img2, img1

            # img1_warp = tfa.image.dense_image_warp(img1, flow)
            img1_warp, mask_boundary_img1 = warp_back(img1, flow)
            mask2 = calculate_luminance_mask(img1_warp, img2, mask)
            TODO - mask_boundary_1, mask2

            feat1, output_img1 = style_model(img1)
            feat2, output_img2 = style_model(img2)

            # Calculate flow and mask for feature 1
            resize_size = (feat1.get_shape()[1], feat1.get_shape()[2])
            print(f"resizing to {resize_size}")
            feat1_flow = tf.image.resize(images=flow, size=resize_size)
            feat1_mask = tf.image.resize(images=mask, size=resize_size)

            # feat1_warp = tfa.image.dense_image_warp(
            #     feat1, feat1_flow
            # )  # TODO - replace with final routine
            # temp_feature_loss = sum_mse_loss(feat2, feat1_warp)
            #
            # # mastk_feat1 = get_mask() # TODO - implement this

            # temp_feature_loss
            # temp_feature_loss = sum_mse_loss(feat2, feat1) # TODO - change to feat_1_warp

            # return temporary_feature_loss


if __name__ == "__main__":
    # build_toy_model()
    args = parser.parse_args()
    print(type(args))

    cnn_normalization_mean = tf.constant([0.485, 0.456, 0.406])
    cnn_normalization_std = tf.constant([0.229, 0.224, 0.225])

    # read in style image (weight, height)
    style_img = Image.open(args.style_name)
    style_img = style_img.resize((args.width, args.height), Image.BILINEAR)
    style_img = tf.convert_to_tensor(style_img)
    # style_img = tf.transpose(style_img, (2, 0, 1)) # no need to transpose since tf tensors are ordered differently
    print(style_img)

    style_model = ReCoNet()
    # loss_model = VGG16(weights="imagenet", classifier_activation=)
    loss_model = VGG16()

    # mse_loss = losses.MeanSquaredError(reduction=losses.Reduction.SUM_OVER_BATCH_SIZE, name="mse")
    sum_mse_loss = losses.MeanSquaredError(reduction="sum_over_batch_size", name="mse")
    mse_loss = losses.MeanSquaredError(reduction="none", name="summed_mse")
    optimizer = optimizers.Adamax(learning_rate=args.lr)

    train_data = MPIDataSet(Path(args.path).joinpath("training"), args)
    test_data = MPIDataSet(Path(args.path).joinpath("test"), args)

    signature = (tf.float32, tf.float32, tf.float32, tf.float32)
    # Convert to dataset
    train_dataset = Dataset.from_generator(
        train_data, output_types=signature, name="train"
    )

    # might want to shuffle, but slow for debugging
    train_dataset = train_dataset.batch(args.batch_size)

    test_dataset = Dataset.from_generator(
        test_data, output_types=signature, name="test"
    )

    # train model
    train(
        args=args,
        train_dataset=train_dataset,
        # test_dataset=test_dataset,
        loss_model=loss_model,
        mse_loss=mse_loss,
        sum_mse_loss=sum_mse_loss,
        optimizer=optimizer,
        style_model=style_model,
        norm_mean=cnn_normalization_mean,
        norm_std=cnn_normalization_std,
    )

    # print("train len:", len(train_dataset))
    # print("test len:", len(test_dataset))
