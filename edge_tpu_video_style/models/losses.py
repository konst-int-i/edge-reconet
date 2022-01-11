import tensorflow as tf
from dataclasses import dataclass

from typing import Iterable
from models.utils import warp_back, get_luminance_grayscale, calculate_luminance_mask


@dataclass
class ReCoNetLoss:
    """Write a class, they said. It'll be fun, they said"""

    alpha: float
    beta: float
    gamma: float
    lambda_f: float
    lambda_o: float

    _content_loss: float = 0.0
    _style_loss: float = 0.0
    _feature_temporal_loss: float = 0.0
    _output_temporal_loss: float = 0.0
    _total_variation: float = 0.0

    def __call__(
        self,
        current_vgg_out,
        current_vgg_in,
        previous_vgg_out,
        style_gram_matrices,
        current_output_frame,
        previous_output_frame,
        current_input_frame,
        previous_input_frame,
        current_feature_maps,
        previous_feature_maps,
        reverse_optical_flow,
        occlusion_mask,
    ) -> float:
        """Calculate the full loss for the ReCoNet model

        Args:
            current_vgg_out (tf.Tensor): The vgg output for the current frame
            current_vgg_in (tf.Tensor): The vgg input for the current frame
            previous_vgg_out (tf.Tensor): The vgg output for the previous frame
            style_gram_matrices (Iterable[tf.Tensor]): A list of style gram matricies
            current_output_frame (tf.Tensor): The current output frame of the ReCoNet model
            previous_output_frame (tf.Tensor): The previous output frame of the ReCoNet model
            current_input_frame (tf.Tensor): The current input frame of the ReCoNet model
            previous_input_frame (tf.Tensor): The previous input frame of the ReCoNet model
            current_feature_maps (tf.Tensor): The current feature maps from the ReCoNet encoder
            previous_feature_maps (tf.Tensor): The previous feature maps from the ReCoNet encoder
            reverse_optical_flow (tf.Tensor): The optical flow from frame t to frame t - 1
            occlusion_mask (tf.Tensor): The occlusion mask

        Returns:
            float: The loss for the reconet model
        """

        content_loss_ = self.alpha * content_loss(
            current_vgg_out[2], current_vgg_in[2]
        ) + self.alpha * content_loss(previous_vgg_out[2], current_vgg_out[2])
        # TODO - shouldn't this be 1 in TF?

        style_loss_ = self.beta * style_loss(
            current_vgg_out, style_gram_matrices
        ) + self.beta * style_loss(previous_vgg_out, style_gram_matrices)

        total_variation_ = self.gamma * total_variation(
            current_output_frame
        ) + self.gamma * total_variation(previous_output_frame)

        feature_temporal_loss_ = self.lambda_f * feature_temporal_loss(
            current_feature_maps,
            previous_feature_maps,
            reverse_optical_flow,
            occlusion_mask,
        )

        output_temporal_loss_ = self.lambda_o * output_temporal_loss(
            current_input_frame,
            previous_input_frame,
            current_output_frame,
            previous_output_frame,
            reverse_optical_flow,
            occlusion_mask,
        )

        self._content_loss = content_loss_
        self._style_loss = style_loss_
        self._total_variation = total_variation_
        self._feature_temporal_loss = feature_temporal_loss_
        self._output_temporal_loss = output_temporal_loss_

        return sum([
            content_loss_,
            style_loss_,
            total_variation_,
            feature_temporal_loss_,
            output_temporal_loss_,
        ])

    def __repr__(self):
        loss_str = f"""{self._content_loss=}, 
                       {self._style_loss=}, 
                       {self._total_variation=}, 
                       {self._feature_temporal_loss=},
                       {self._output_temporal_loss=}"""
        return loss_str


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    # TODO - check if these are the right shapes?
    return result / (num_locations)


def temporal_feature_loss(feat1_warp, feat2):
    return tf.square(feat2 - feat1_warp)


def content_loss(content_feature_maps, style_feature_maps):
    b, w, h, c = content_feature_maps.shape
    # w, h, c = content_feature_maps.shape
    return tf.reduce_sum(tf.square(content_feature_maps - style_feature_maps)) / (
        c * w * h
    )


def style_loss(content_feature_maps, style_gram_matrices):
    return tf.reduce_sum([
        tf.reduce_sum(tf.math.square(gram_matrix(content) - style)) for content, style in zip(content_feature_maps, style_gram_matrices)
    ])


def feature_temporal_loss(
    current_feature_maps, previous_feature_maps, reverse_optical_flow, occlusion_mask
):
    b, w, h, c = current_feature_maps.shape
    reverse_optical_flow_resized = tf.image.resize(images=reverse_optical_flow, size=(w, h))
    occlusion_mask_resized = tf.image.resize(images=occlusion_mask, size=(w, h))
    warp_previous, _ = warp_back(previous_feature_maps, reverse_optical_flow_resized)
    feature_maps_diff = current_feature_maps - warp_previous
    loss = tf.reduce_sum(tf.square(occlusion_mask_resized * feature_maps_diff)) / (c * h * w)
    return loss


def output_temporal_loss(
    current_input_frame,
    previous_input_frame,
    current_output_frame,
    previous_output_frame,
    reverse_optical_flow,
    occlusion_mask,
):
    input_diff = current_input_frame - warp_back(
        previous_input_frame, reverse_optical_flow
    )
    output_diff = current_output_frame - warp_back(
        previous_output_frame, reverse_optical_flow
    )
    print(input_diff.get_shape())
    # get rgb values from input_diff
    red_coef = 0.2126
    green_coef = 0.7152
    blue_coef = 0.0722
    # luminance_input_diff = tf.expand_dims(get_luminance_grayscale(input_diff))
    luminance_input_diff = get_luminance_grayscale(
        input_diff, red_coef, green_coef, blue_coef
    )
    luminance_input_diff = tf.expand_dims(luminance_input_diff, axis=3)
    b, w, h, c = current_input_frame.shape
    loss = tf.reduce_sum(
        tf.math.square((occlusion_mask * (output_diff - luminance_input_diff)))
    ) / (h * w)
    return loss


def total_variation(im):
    return tf.reduce_sum(
        tf.math.abs(im[:, :, :-1, :] - im[:, :, 1:, :])
    ) + tf.reduce_sum(tf.math.abs(im[:, :-1, :, :] - im[:, 1:, :, :]))
