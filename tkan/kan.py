
import tensorflow as tf
from tensorflow.keras.initializers import Initializer, HeUniform
from tensorflow.keras.layers import Layer, Dropout

class GridInitializer(Initializer):
    """Initializes a grid for use in B-spline calculations within KANLinear layer.

    Args:
        grid_range (list of float): The min and max range values for the grid.
        grid_size (int): The number of intervals in the grid.
        spline_order (int): The order of the spline transformation.
    """
    def __init__(self, grid_range, grid_size, spline_order):
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order

    def __call__(self, shape, dtype=None):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = tf.linspace(
            start=-self.spline_order * h + self.grid_range[0],
            stop=(self.grid_size + self.spline_order) * h + self.grid_range[0],
            num=self.grid_size + 2 * self.spline_order + 1
        )
        grid = tf.tile(tf.expand_dims(grid, 0), [shape[0], 1])
        return grid

    def get_config(self):
        return {
            "grid_range": self.grid_range,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order
        }

class KANLinear(Layer):
    """Custom Keras layer that implements Kernel Additive Networks (KAN) with B-spline transformations.

    This layer supports tensors with more than two dimensions, treating the last dimension as features
    and applies transformations uniformly across any preceding dimensions (e.g., time steps in sequences).

    Do not need in_features, computed at build time

    Args:
        units (int): Number of output units.
        grid_size (int, optional): Size of the grid for spline calculations. Defaults to 5.
        spline_order (int, optional): Order of the spline transformations. Defaults to 3.
        scale_noise (float, optional): Scaling factor for noise. Defaults to 0.1.
        scale_base (float, optional): Base scaling factor. Defaults to 1.0.
        scale_spline (float, optional): Spline scaling factor. Defaults to 1.0.
        base_activation (str, optional): Activation function to use. Defaults to 'silu'.
        grid_eps (float, optional): Epsilon value for grid adjustments. Defaults to 0.02.
        grid_range (list of float, optional): Range of values for the grid. Defaults to [-1, 1].
    """
    def __init__(
        self,
        units,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation='silu',
        grid_eps=0.02,
        grid_range=[-1, 1],
        dropout = 0.,
        **kwargs
    ):
        super(KANLinear, self).__init__(**kwargs)
        self.in_features = None
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.base_activation = getattr(tf.nn, base_activation)
        self.grid_eps = grid_eps
        self.grid_range = grid_range

        self.dropout = Dropout(dropout)
    
    def build(self, input_shape):
        super(KANLinear, self).build(input_shape)
        self.in_features = input_shape[-1]
        self.other_dims = input_shape[1:-1]
        self.grid = self.add_weight(
            name="grid",
            shape=[self.in_features, self.grid_size + 2 * self.spline_order + 1],
            initializer=GridInitializer(self.grid_range, self.grid_size, self.spline_order),
            trainable=False
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=[self.units, self.in_features],
            initializer=HeUniform
        )

        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=[self.units, self.in_features * (self.grid_size + self.spline_order)],
            initializer=HeUniform
        )

    def call(self, x):
        base_output = tf.matmul(self.base_activation(x), self.base_weight, transpose_b=True)
        spline_output = tf.matmul(
            self.b_splines(x),
            self.spline_weight,
            transpose_b=True
        )
        return self.dropout(base_output) + self.dropout(spline_output)
            
    def b_splines(self, x):
        batch_size = tf.shape(x)[0]
        x_expanded = tf.expand_dims(x, -1)  
        
        
        grid = self.grid  
        grid_expanded = tf.expand_dims(grid, 0)  
        
        grid_expanded = tf.broadcast_to(grid_expanded, (batch_size, self.in_features, self.grid.shape[1]))

        done_dims = []
        for dim in self.other_dims:
            grid_expanded = tf.expand_dims(grid_expanded, 1) 
            grid_expanded = tf.broadcast_to(grid_expanded, (batch_size, dim, *done_dims, self.in_features, self.grid.shape[1]))
            done_dims.append(dim)

        bases = tf.cast((x_expanded >= grid_expanded[..., :-1]) & (x_expanded < grid_expanded[..., 1:]), x.dtype)

        for k in range(1, self.spline_order + 1):
            left_denominator = grid[:, k:-1] - grid[:, :-(k + 1)] + 1e-10
            right_denominator = grid[:, k + 1:] - grid[:, 1:-k] + 1e-10
            
            left = (x_expanded - grid_expanded[..., :-(k + 1)]) / tf.expand_dims(left_denominator, 0)
            right = (grid_expanded[..., k + 1:] - x_expanded) / tf.expand_dims(right_denominator, 0)
    
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        bases = tf.reshape(bases, (batch_size, *bases.shape[1:-2], -1))
        return bases

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)