import tensorflow as tf
from tensorflow.data import Dataset
import PIL 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_addons as tfa

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


def feature_temporal_loss(
    current_feature_maps, previous_feature_maps, reverse_optical_flow, occlusion_mask
):
    loss = 0
    for feature_map, prev_fm in zip(current_feature_maps.values(), previous_feature_maps.values()):
        b, w, h, c = feature_map.shape
        #reverse_optical_flow_resized = tf.image.resize(
        #    images=reverse_optical_flow, size=(w, h)
        #)
        #occlusion_mask_resized = tf.image.resize(images=occlusion_mask, size=(w, h))
        warp_previous, warp_mask = warp_back(
            prev_fm, reverse_optical_flow
        )
        feature_maps_diff = feature_map - warp_previous
        loss += tf.reduce_sum(
            tf.square(occlusion_mask* feature_maps_diff * warp_mask)
        ) / (c * h * w)
    return loss




def warp_back(image: tf.Tensor, flow: tf.Tensor):
    """Calculates the inverse warping from frame t to frame t - 1
    TODO: Test this method!
    Args:
        image (tf.Tensor): The image tensor at frame t
        flow (tf.Tensor): The optical flow associated with frames t - 1 to t
    Returns:
        tf.Tensor: The inversely-flowed tensor calculated through bilinear
        interpolation
    """
    batch_size, height, width, channels = image.shape

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid + flow

    assert (
        query_points_on_grid.shape[3] == 2
    ), f"Wrong size grid, {query_points_on_grid.shape}"

    # Scale back to [-1, 1]

    width_mult = max(width - 2.0, 0.0)
    height_mult = max(height - 2.0, 0.0)
    multiplier = tf.constant([width_mult, height_mult], dtype=tf.float32)

    query_points_on_grid /= multiplier

    query_points_flattened = tf.reshape(
        query_points_on_grid, [batch_size, height * width, 2]
    )

    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = tfa.image.interpolate_bilinear(image, query_points_flattened)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

    mask = tfa.image.interpolate_bilinear(tf.ones(image.shape), query_points_flattened)
    mask = tf.reshape(mask, [batch_size, height, width, channels])
    mask = tf.where(mask < 0.9999, 0.0, 1.0)

    return interpolated * mask, mask


# weightings from paper get_mask_2


def get_luminance_grayscale(image: tf.Tensor, *luminance_coefs) -> tf.Tensor:
    assert len(luminance_coefs) == 3
    r_coef, g_coef, b_coef = luminance_coefs

    luminance_grayscale = (
        r_coef * image[:, :, :, 0]
        + g_coef * image[:, :, :, 1]
        + b_coef * image[:, :, :, 2]
    )

    return luminance_grayscale


def calculate_luminance_mask(
    current_im: tf.Tensor, previous_im: tf.Tensor, mask: tf.Tensor
) -> tf.Tensor:
    """Get a mask to retain the unchanged places of the current and previous frames,
    the mask preserves still pixels while occluding changed ones
    TODO: Test this method!
    Args:
        current_im (tf.Tensor): The current image
        previous_im (tf.Tensor): The previous image
        mask (tf.Tensor): Occlusion mask
    Returns:
        tf.Tensor: The mask of unchanged places of the current and previous frames
    """
    red_coef = 0.2126
    green_coef = 0.7152
    blue_coef = 0.0722

    image_luminance = get_luminance_grayscale(
        current_im, red_coef, green_coef, blue_coef
    )
    previous_luminance = get_luminance_grayscale(
        previous_im, red_coef, green_coef, blue_coef
    )

    image_luminance = tf.expand_dims(image_luminance, axis=-1)
    previous_luminance = tf.expand_dims(previous_luminance, axis=-1)

    counter_mask = tf.abs(image_luminance - previous_luminance)

    counter_mask = tf.where(counter_mask < 0.05, 0.0, 1.0)
    counter_mask = mask - counter_mask
    counter_mask = tf.where(counter_mask < 0, 0.0, 1.0)

    return counter_mask



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



def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)



def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

content_image = load_img(content_path)
style_image = load_img(style_path)

vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
  print(layer.name)


def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(tf.expand_dims(style_image, axis=-1)*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())


style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
print(content_image.shape)

image = tf.Variable(content_image)


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight=1e-2
content_weight=1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss, outputs['style']


def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


import time
start = time.time()

epochs = 1
steps_per_epoch = 10
total_variation_weight=30



class ReCoNetLoss:
    """Write a class, they said. It'll be fun, they said"""

    alpha: float
    beta: float
    gamma: float
    lambda_f: float
    lambda_o: float
    temp_output_scale: float
    temp_feat_scale: float

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
        ):
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

        style_loss_ = self.beta * style_loss(
            current_vgg_out, style_gram_matrices
        ) + self.beta * style_loss(previous_vgg_out, style_gram_matrices)

        total_variation_ = self.gamma * total_variation(
            current_output_frame
        ) + self.gamma * total_variation(previous_output_frame)

        feature_temporal_loss_ = (
            self.lambda_f
            * self.temp_feat_scale
            * feature_temporal_loss(
                current_feature_maps,
                previous_feature_maps,
                reverse_optical_flow,
                occlusion_mask,
            )
        )

        output_temporal_loss_ = (
            self.lambda_o
            * self.temp_output_scale
            * output_temporal_loss(
                current_input_frame,
                previous_input_frame,
                current_output_frame,
                previous_output_frame,
                reverse_optical_flow,
                occlusion_mask,
            )
        )

        self._content_loss = content_loss_
        self._style_loss = style_loss_
        self._total_variation = total_variation_
        self._feature_temporal_loss = feature_temporal_loss_
        self._output_temporal_loss = output_temporal_loss_

        return [
            content_loss_,
            style_loss_,
            total_variation_,
            feature_temporal_loss_,
            output_temporal_loss_,
        ]

    def __repr__(self):
        loss_str = f"""content_loss={self._content_loss}, 
                       style_loss={self._style_loss}, 
                       total_variation={self._total_variation}, 
                       feature_temporal_loss={self._feature_temporal_loss},
                       output_temporal_loss={self._output_temporal_loss}"""
        return loss_str


from preprocessing import MPIDataSet
from models.layers import ReCoNet

reconet = ReCoNet()

class Foo:
    ...

args = Foo()
args.width=512
args.height=422
args.batch_size=1
from pathlib import Path

train_data = MPIDataSet(Path("MPI-Sintel-complete/training"), args)
signature = (tf.float32, tf.float32, tf.float32, tf.float32)
train_dataset = Dataset.from_generator(
    train_data, output_types=signature, name="train"
)

# might want to shuffle, but slow for debugging
train_dataset = train_dataset.batch(args.batch_size)

@tf.function()
def train_step(image, previous_img, flow, mask):
  reverse_optical_flow = warp_back(image, flow)
  with tf.GradientTape() as tape:
    output_frame = reconet(image)
    previous_output_frame = reconet(previous_img)
    outputs = extractor(image)
    previous_outputs = extractor(previous_img)
    loss, feature_maps = style_content_loss(outputs)
    pscl, prev_feature_maps = style_content_loss(previous_outputs)
    loss += pscl
    loss += total_variation_weight*tf.image.total_variation(image)
    loss += feature_temporal_loss(feature_maps, prev_feature_maps, reverse_optical_flow, mask)
    loss += output_temporal_loss(image, previous_img, output_frame, previous_output_frame, reverse_optical_flow, occlusion_mask)
  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))
  return outputs

for sample in train_dataset:
    current_img, prev_img, flow, mask = sample
    img = tf.Variable(tf.transpose(current_img, (0, 2, 1, 3)))
    previous_img = tf.Variable(tf.transpose(prev_img, (0, 2, 1, 3)))
    flow = tf.Variable(tf.transpose(flow, (0, 2, 1, 3)))
    mask = tf.Variable(tf.transpose(mask, (0, 2, 1, 3)))
    plt.imshow(img[0,:,:,:])
    plt.show()
    for _ in range(10):
        out = train_step(img, previous_img, flow, mask)
        plt.imshow(out[0,:,:,:])
        plt.show()
    plt.imshow(img[0,:,:,:])
    plt.show()

