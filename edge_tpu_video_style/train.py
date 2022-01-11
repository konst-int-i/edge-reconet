import tensorflow as tf
from edge_tpu_video_style.models.losses import temporal_feature_loss
from edge_tpu_video_style.models.utils import calculate_luminance_mask
from models.reconet import make_reconet
from models.layers import ReCoNet # Needs to be moved into reconet.py
from preprocessing import MPIDataSet
from models.losses import ReCoNetLoss, gram_matrix

from pathlib import Path

from models.utils import warp_back, calculate_luminance_mask


class Foo:
    ...


@tf.function()
def train_step(sample):
    reconet = ReCoNet()
    loss_fn = ReCoNetLoss()
    vgg = ...
    style = ...
    with tf.GradientTape() as tape:
        current_frame, previous_frame, mask, flow = sample
        current_frame_anti_warp = warp_back(current_frame, flow)
        occlusion_mask = calculate_luminance_mask(current_frame_anti_warp, previous_frame, mask)
        
        current_frame_features, current_frame_output = reconet(current_frame)
        previous_frame_features, previous_frame_output = reconet(previous_frame)

        _, feature_width, feature_height, _ = current_frame_features.shape
        current_feature_flow = tf.image.resize(images=flow, size=(feature_width, feature_height))
        current_feature_warp, _ = warp_back(current_frame_features, current_feature_flow)

        current_vgg_in = vgg(current_frame)
        current_vgg_out = vgg(current_frame_output)
        previous_vgg_in = vgg(previous_frame)
        previous_vgg_out = vgg(previous_frame_output)

        style_gram_matricies = [gram_matrix(x) for x in vgg(style)]

        loss = loss_fn(
            current_vgg_out=current_vgg_out, 
            current_vgg_in=current_vgg_in, 
            previous_vgg_out=previous_vgg_out,
            style_gram_matrices=style_gram_matricies, 
            current_output_frame=current_frame_output, 
            previous_output_frame=previous_frame_output,
            current_input_frame=current_frame, 
            previous_input_frame=previous_frame, 
            current_feature_maps=current_frame_features,
            previous_feature_maps=previous_frame_features, 
            reverse_optical_flow=current_feature_warp, #Also unsure
            occlusion_mask=occlusion_mask  # Unsure?
        )

    return tape.gradient(loss, [current_frame, previous_frame, mask, flow])

# def build_model() -> tf.keras.Model:
#     loss = ReCoNetLoss()
#     inputs, outputs = build_reconet()
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name="reconet")

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(3e-4),
#         loss=loss
#     )
#     return model



def main():
    args = Foo()
    args.width = 512
    args.height = 216
    args.path = "MPI-Sintel-complete"
    train_data = MPIDataSet(Path(args.path).joinpath("training"), args)
    test_data = MPIDataSet(Path(args.path).joinpath("test"), args)

    signature = (tf.float32, tf.float32, tf.float32, tf.float32)

