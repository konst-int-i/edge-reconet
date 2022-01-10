import tensorflow as tf
import tensorflow_addons as tfa

def save_model(quantised, name='style_transfer.tflite'):
    with open(f'saved_models/{name}', 'wb') as f:
        f.write(quantised)


def warp_back(image: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
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
    
    assert query_points_on_grid.shape[3] == 2, f"Wrong size grid, {query_points_on_grid.shape}"

    # Scale back to [-1, 1]
    query_points_on_grid = query_points_on_grid[:, :, :, 0] / tf.max(width-1, 1)-1.0
    query_points_on_grid = query_points_on_grid[:, :, :, 1] / tf.max(height-1, 1)-1.0

    query_points_flattened = tf.reshape(
        query_points_on_grid, [batch_size, height * width, 2]
    )

    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = tfa.image.interpolate_bilinear(image, query_points_flattened)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

    mask = tfa.image.interpolate_bilinear(tf.ones(image.shape), query_points_flattened)
    mask = tf.reshape(mask, [batch_size, height, width, channels])
    mask = tf.where(mask < 0.9999, 0, 1)

    return interpolated * mask, mask 


# weightings from paper get_mask_2

def _get_luminance_grayscale(image, *luminance_coefs):
    assert len(luminance_coefs) == 3
    r_coef, g_coef, b_coef = luminance_coefs

    luminance_grayscale = r_coef * image[:, :, :, 0] + \
                          g_coef * image[:, :, :, 1] + \
                          b_coef * image[:, :, :, 2]

    return luminance_grayscale


def get_mask(current_im: tf.Tensor, previous_im: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Get a mask to retain the unchanged places of the current and previous frames,
    the mask preserves still pixels while occluding changed ones
    TODO: Test this method!

    Args:
        current_im (tf.Tensor): Optical flow current_im
        previous_im (tf.Tensor): The previous current_im
        mask (tf.Tensor): Occlusion mask

    Returns:
        tf.Tensor: [description]
    """
    red_coef = 0.2126
    green_coef = 0.7152
    blue_coef = 0.0722

    image_luminance = _get_luminance_grayscale(current_im, red_coef, green_coef, blue_coef)
    previous_luminance = _get_luminance_grayscale(previous_im, red_coef, green_coef, blue_coef)

    image_luminance = tf.expand_dims(image_luminance)
    previous_luminance = tf.expand_dims(previous_luminance)

    counter_mask = tf.abs(image_luminance - previous_luminance)

    counter_mask = tf.where(counter_mask < 0.05, 0, 1)
    counter_mask = mask - counter_mask
    counter_mask = tf.where(counter_mask < 0, 0, 1)

    return counter_mask