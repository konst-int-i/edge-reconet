import argparse
import os
from models.reconet import build_reconet
from models.utils import save_model, warp_back, calculate_luminance_mask
from postprocessing.quantisation import quantise_model
from edge_tpu_video_style.utils.parser import parser
from models.layers import ReCoNet
from models.losses import *
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
from dataclasses import dataclass
from typing import Iterable

from models.losses import temporal_feature_loss


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
            img_previous, img_current, mask, flow = sample
            # optical flow in dataset is opposite
            img_current, img_previous = img_previous, img_current

            current_inverse_warp, transition_mask_boundary = warp_back(
                img_current, flow
            )
            luminance_mask = calculate_luminance_mask(
                current_inverse_warp, img_current, mask
            )

            feat_previous, output_img_previous = style_model(img_previous)
            feat_current, output_img_current = style_model(img_current)

            # Calculate flow and mask for feature 1
            resize_size = (feat_previous.get_shape()[1], feat_previous.get_shape()[2])
            print(f"resizing to {resize_size}")
            feat_previous_flow = tf.image.resize(images=flow, size=resize_size)
            feat_previous_mask = tf.image.resize(images=mask, size=resize_size)

            feat_previous_inv_warp, feat_previous_mask_boundary = warp_back(
                feat_previous, feat_previous_flow
            )
            # temp_feature_loss = temporal_feature_loss(feat_previous_warp, feat_current)

            # ReCoNetLoss()
            print(f"feat2 shape: ", feat_current.get_shape())
            print(f"feat1_flow shape: ", feat_previous_flow.get_shape())
            print(f"feat1_warp_shape: ", feat_previous_inv_warp.get_shape())
            print(f"Mask boundary_img1: ", transition_mask_boundary.get_shape())
            feat_previous_mask = calculate_luminance_mask(
                feat_previous_inv_warp, feat_current, feat_previous_mask
            )

            # temporal feature loss
            temp_feature_loss = feature_temporal_loss(
                current_feature_maps=feat_current,
                previous_feature_maps=feat_previous,
                reverse_optical_flow=feat_previous_flow,
                occlusion_mask=feat_previous_mask,
            )
            print("Temporal feature loss", temp_feature_loss)

            # get output temporal loss
            # temp_ouptput_loss = output_temporal_loss(
            #     current_input_frame=img_current,
            #     previous_input_frame=img_previous,
            #     current_output_frame=output_img_current,
            #     previous_output_frame=output_img_previous,
            #     reverse_optical_flow=flow,
            #     occlusion_mask=mask
            # )
            # print("Temporal output loss", temp_ouptput_loss)

            # get content feature maps
            # normalize
            img_current_norm = normalization(output_img_current)
            img_previous_norm = normalization(output_img_previous)

            # pass through vgg
            # vgg_output_current = loss_model(img_current_norm)
            # vgg_output_previous = loss_model(img_previous_norm)

            # print(vgg_output_current.get_shape())
            # print(vgg_output_previous.get_shape())
            # vgg_input_current =


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
    from keras.utils import losses_utils

    # mse_loss = losses.MeanSquaredError(reduction=losses.Reduction.SUM_OVER_BATCH_SIZE, name="mse")
    sum_mse_loss = losses.MeanSquaredError(reduction="sum_over_batch_size", name="mse")
    mse_loss = losses.MeanSquaredError(
        reduction=losses_utils.ReductionV2.NONE, name="summed_mse"
    )
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
