import tensorflow as tf
from tensorflow.data import Dataset

from edge_tpu_video_style.preprocessing.read_mpi import read_style_image
import tensorflow as tf
from models.layers import ReCoNet, ReconetNorm, ReconetUnnorm
from preprocessing import MPIDataSet
from tensorflow.keras import optimizers
from tqdm import tqdm
from models.layers import Normalization
from models.losses import ReCoNetLoss, gram_matrix
from utils.parser import parser
from tensorflow.keras.applications.vgg16 import preprocess_input

from pathlib import Path

from models.utils import warp_back, calculate_luminance_mask


NORM_MEAN = tf.constant([0.485, 0.456, 0.406])
NORM_STD = tf.constant([0.229, 0.224, 0.225])
normalization = Normalization(NORM_MEAN, NORM_STD)


def vgg_prep(img):
    return preprocess_input(255 * img)


def vgg_postprocess(images):
    return [img / 255 for img in images]


def call_vgg(img, vgg):
    return vgg_postprocess(vgg(vgg_prep(img)))


def vgg_layers():
    """Creates a vgg model that returns a list of intermediate output values."""

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
    ]

    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in style_layers]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def vgg_norm(image):
    return normalization(image)


# @tf.function(autograph=True)
def train_step(args, sample, style, reconet, vgg):
    loss_fn = ReCoNetLoss(
        args["ALPHA"],
        args["BETA"],
        args["GAMMA"],
        args["LAMBDA_F"],
        args["LAMBDA_O"],
        args["temp_output_scale"],
        args["temp_feature_scale"],
    )
    norm = ReconetNorm()
    unnorm = ReconetUnnorm()

    with tf.GradientTape() as tape:
        tape.watch(reconet.trainable_weights)

        current_frame, previous_frame, flow, motion_boundaries = sample
        inverse_warp, reverse_motion_boundaries = warp_back(current_frame, flow)
        occlusion_mask = calculate_luminance_mask(
            inverse_warp, previous_frame, motion_boundaries
        )

        current_frame_features, current_frame_output = reconet(norm(current_frame))
        previous_frame_features, previous_frame_output = reconet(norm(previous_frame))

        _, feature_width, feature_height, _ = current_frame_features.shape
        current_feature_flow = tf.image.resize(
            images=flow, size=(feature_width, feature_height)
        )
        current_feature_warp, _ = warp_back(
            current_frame_features, current_feature_flow
        )

        with tape.stop_recording():

            # Unonrmalize from reconet and apply VGG-specific norm
            current_vgg_out = call_vgg(unnorm(current_frame_output), vgg)
            previous_vgg_out = call_vgg(unnorm(previous_frame_output), vgg)
            current_vgg_in = call_vgg(current_frame, vgg)
            previous_vgg_in = call_vgg(previous_frame, vgg)
            style_gram_matrices = [gram_matrix(x) for x in call_vgg(style, vgg)]

        loss = loss_fn(
            current_vgg_out=current_vgg_out,
            current_vgg_in=current_vgg_in,
            previous_vgg_out=previous_vgg_out,
            style_gram_matrices=style_gram_matrices,
            current_output_frame=current_frame_output,
            previous_output_frame=previous_frame_output,
            current_input_frame=norm(current_frame),
            previous_input_frame=norm(previous_frame),
            current_feature_maps=current_frame_features,
            previous_feature_maps=previous_frame_features,
            reverse_optical_flow=flow,  # Also unsure
            occlusion_mask=occlusion_mask,  # Unsure?
        )

    print(f"{loss=}")
    print(loss_fn)

    gradients = tape.gradient(loss, reconet.trainable_weights)
    return (gradients, reconet.trainable_weights)


def train_loop(args, train_data, optimizer, style, reconet, vgg) -> ReCoNet:

    epochs = 1 if args.debug else args.epochs
    print(f"{epochs=}")

    for epoch in range(epochs):
        data_bar = tqdm(train_data)
        for id, sample in enumerate(data_bar):

            if args.debug and id == 2:
                break

            gradients, reconet_vars = train_step(
                vars(args), sample, style, reconet, vgg
            )
            optimizer.apply_gradients(zip(gradients, reconet_vars))

    return reconet


def main():
    # Load in MPI train data
    args = parser.parse_args()
    train_data = MPIDataSet(Path(args.path).joinpath("training"), args)

    # convert to tf dataset & batch
    signature = (tf.float32, tf.float32, tf.float32, tf.float32)
    train_dataset = Dataset.from_generator(
        train_data, output_types=signature, name="train"
    )

    # might want to shuffle, but slow for debugging
    train_dataset = train_dataset.batch(args.batch_size)

    optimizer = optimizers.Adamax(learning_rate=args.lr)

    # read & process style image
    style_img = read_style_image(args)

    reconet = ReCoNet()
    vgg = vgg_layers()

    trained_reconet = train_loop(
        args, train_dataset, optimizer, style_img, reconet, vgg
    )

    trained_reconet.save(f"saved_models/{args.model_name}")


if __name__ == "__main__":
    main()
