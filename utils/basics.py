import tensorflow.compat.v1 as tf
import numpy as np


def interpolate_bilinear(grid, query_points, indexing="ij"):
    """
    :param grid:
    :param query_points:
    :param indexing:
    :return:
    """
    grid = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)

    # grid shape checks
    grid_static_shape = grid.shape
    grid_shape = tf.shape(grid)
    if grid_static_shape.dims is not None:
        if len(grid_static_shape) != 4:
            raise ValueError("Grid must be 4D Tensor")
        if grid_static_shape[1] is not None and grid_static_shape[1] < 2:
            raise ValueError("Grid height must be at least 2.")
        if grid_static_shape[2] is not None and grid_static_shape[2] < 2:
            raise ValueError("Grid width must be at least 2.")
    else:
        pass

    # query_points shape checks
    query_static_shape = query_points.shape
    # print(query_points.shape)
    query_shape = tf.shape(query_points)
    if query_static_shape.dims is not None:
        if len(query_static_shape) != 3:
            raise ValueError("Query points must be 3 dimensional.")
        query_hw = query_static_shape[2]
        if query_hw is not None and query_hw != 2:
            raise ValueError("Query points last dimension must be 2.")
    else:
        pass

    batch_size, height, width, channels = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        grid_shape[3],
    )

    num_queries = query_shape[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

    for i, dim in enumerate(index_order):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = grid_shape[i + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = tf.constant(0.0, dtype=query_type)
        floor = tf.math.minimum(
            tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
        )
        int_floor = tf.cast(floor, tf.dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)

    flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
    batch_offsets = tf.reshape(
        tf.range(batch_size) * height * width, [batch_size, 1]
    )

    # This wraps tf.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using tf.gather_nd.
    def gather(y_coords, x_coords, name):
        with tf.name_scope("gather-" + name):
            linear_coordinates = batch_offsets + y_coords * width + x_coords
            gathered_values = tf.gather(flattened_grid, linear_coordinates)
            return tf.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    # now, do the actual interpolation
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def warp(teninput, tenflow):
    batch_size, height, width, channels = (
        tf.shape(teninput)[0],
        tf.shape(teninput)[1],
        tf.shape(teninput)[2],
        tf.shape(teninput)[3],
    )
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), tenflow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - tenflow
    query_points_flattened = tf.reshape(
        query_points_on_grid, [batch_size, height * width, 2]
    )
    interpolated = interpolate_bilinear(teninput, query_points_flattened)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated


def quantize_image(image):
    image = np.reshape(image, (image.shape[1], image.shape[2], 3))
    image = tf.convert_to_tensor(image)
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)