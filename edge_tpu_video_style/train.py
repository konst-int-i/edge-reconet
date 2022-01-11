import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from models.losses import temporal_feature_loss
from models.utils import calculate_luminance_mask
# from models.reconet import make_reconet
from models.layers import ReCoNet # Needs to be moved into reconet.py
from preprocessing import MPIDataSet
from models.layers import Normalization
from models.losses import ReCoNetLoss, gram_matrix

from pathlib import Path

from models.utils import warp_back, calculate_luminance_mask


NORM_MEAN = tf.constant([0.485, 0.456, 0.406])
NORM_STD = tf.constant([0.229, 0.224, 0.225])
normalization = Normalization(NORM_MEAN, NORM_STD)


def vgg_layers():
    """ Creates a vgg model that returns a list of intermediate output values."""

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in style_layers]

    model = tf.keras.Model([vgg.input], outputs)
    return model


class Foo:
    ...

def vgg_preprocess(image):
    return normalization(image)

@tf.function()
def train_step(sample, style, reconet, vgg, args):
    loss_fn = ReCoNetLoss(args.ALPHA, args.BETA, args.GAMMA, args.LAMBDA_F, args.LAMBDA_O)

    with tf.GradientTape() as tape:
        current_frame, previous_frame, flow, motion_boundaries = sample
        inverse_warp, reverse_motion_boundaries = warp_back(current_frame, flow)
        occlusion_mask = calculate_luminance_mask(inverse_warp, previous_frame, motion_boundaries)
        
        current_frame_features, current_frame_output = reconet(current_frame)
        previous_frame_features, previous_frame_output = reconet(previous_frame)

        _, feature_width, feature_height, _ = current_frame_features.shape
        current_feature_flow = tf.image.resize(images=flow, size=(feature_width, feature_height))
        current_feature_warp, _ = warp_back(current_frame_features, current_feature_flow)

        current_vgg_in = vgg(vgg_preprocess(current_frame))
        current_vgg_out = vgg(vgg_preprocess(current_frame_output))
        previous_vgg_in = vgg(vgg_preprocess(previous_frame))
        previous_vgg_out = vgg(vgg_preprocess(previous_frame_output))

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
            reverse_optical_flow=flow, #Also unsure
            occlusion_mask=occlusion_mask  # Unsure?
        )

    return tape.gradient(loss, [current_frame, previous_frame, motion_boundaries, flow])


def main():
    args = Foo()
    args.width = 512
    args.height = 216
    args.path = "MPI-Sintel-complete"
    train_data = MPIDataSet(Path(args.path).joinpath("training"), args)
    test_data = MPIDataSet(Path(args.path).joinpath("test"), args)

    signature = (tf.float32, tf.float32, tf.float32, tf.float32)

    im1 = tf.random.uniform(minval=0, maxval=1, shape=(4, 512, 216, 3))
    im2 = tf.random.uniform(minval=0, maxval=1, shape=(4, 512, 216, 3))
    flow = tf.random.uniform(minval=0, maxval=1, shape=(4, 512, 216, 2))
    mask = tf.where(
        tf.random.uniform(minval=0, maxval=1, shape=(4, 512, 216, 1)) > 0.5,
        1., 0.)

    style = tf.random.uniform(minval=0, maxval=1, shape=(4, 512, 216, 3))

    reconet = ReCoNet()
    vgg = vgg_layers()

    args.ALPHA, args.BETA, args.GAMMA, args.LAMBDA_F, args.LAMBDA_O = 1, 1, 1, 1, 1

    sample = (im1, im2, flow, mask)
    train_step(sample, style, reconet, vgg, args)


if __name__=="__main__":
    main()