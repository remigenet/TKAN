import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable(package='tkan', name='FixedSplineActivation')
class FixedSplineActivation(Layer):
    """
    This class defines a custom Keras Layer implementing a fixed spline activation function.
    The activation function modifies the input values using a power law transformation,
    controlled by an exponent that is clamped within a specified range.

    Attributes:
        exponent (tf.Tensor): The exponent used in the power function. This value is cast to float32
            and clamped between -max_exponent and max_exponent.
        max_exponent (tf.Tensor): The maximum allowable absolute value for the exponent.
        epsilon (float): A small constant (from Keras backend) used to maintain numerical stability.

    Methods:
        call(inputs: tf.Tensor) -> tf.Tensor: Applies the activation function to the input tensor.
    """
    def __init__(self, exponent: float = 1.0, max_exponent: float = 9.0, **kwargs) -> None:
        """
        Initializes the FixedSplineActivation layer with the specified exponent and maximum exponent.

        Args:
            exponent (float, optional): The exponent to use in the activation function. Defaults to 1.0.
            max_exponent (float, optional): The maximum allowable absolute value for the exponent.
                Defaults to 9.0.
        """
        super(FixedSplineActivation, self).__init__(**kwargs)
        self.exponent: tf.Tensor = tf.cast(exponent, tf.float32)
        self.epsilon: float = K.epsilon()
        self.max_exponent: tf.Tensor = tf.cast(max_exponent, tf.float32)
        self.exponent: tf.Tensor = tf.clip_by_value(self.exponent, -self.max_exponent, self.max_exponent)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies the power activation function to the inputs, clamping the input values to ensure
        numerical stability before applying the exponent.

        Args:
            inputs (tf.Tensor): Input tensor to be transformed by the activation function.

        Returns:
            tf.Tensor: The transformed tensor, with the same shape as the input.
        """
        inputs_safe: tf.Tensor = tf.clip_by_value(inputs, self.epsilon, 1.0)
        return tf.pow(inputs_safe, self.exponent)

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A Python dictionary containing the layer configuration, ensuring that all 
            initialization parameters are included for correct layer reconstruction upon loading.
        """
        config = super(FixedSplineActivation, self).get_config()
        config.update({
            'exponent': self.exponent.numpy(),  # Convert to native Python type for serialization
            'max_exponent': self.max_exponent.numpy(),
        })
        return config

@tf.keras.utils.register_keras_serializable(package='tkan', name='PowerSplineActivation')
class PowerSplineActivation(Layer):
    """
    A custom Keras Layer implementing a power spline activation function with trainable exponent and bias.
    The activation function transforms the input values using a power law modification, where the exponent can be
    adjusted during training.

    Attributes:
        initial_exponent (tf.Tensor): The initial value of the exponent.
        epsilon (float): A small constant to ensure numerical stability.
        max_exponent (tf.Tensor): The maximum allowed absolute value for the exponent.
        trainable (bool): Flag to determine whether the exponent and bias are trainable.

    Methods:
        build(input_shape): Initializes the layer's weights.
        call(inputs): Applies the activation function to the inputs.
        get_config(): Returns the configuration of the layer.
    """
    def __init__(self, initial_exponent: float = 1.0, epsilon: float = 1e-7, max_exponent: float = 9.0, trainable: bool = True, **kwargs) -> None:
        """
        Initializes the PowerSplineActivation layer with specified initial exponent, epsilon for stability,
        maximum exponent, and whether the parameters are trainable.

        Args:
            initial_exponent (float): The initial value of the exponent for the power function.
            epsilon (float): Small constant to avoid division by zero or log of zero.
            max_exponent (float): Maximum allowable absolute value for the exponent.
            trainable (bool): If True, allows the exponent and bias to be trained.
        """
        super(PowerSplineActivation, self).__init__(**kwargs)
        self.initial_exponent: tf.Tensor = tf.cast(initial_exponent, tf.float32)
        self.epsilon: float = epsilon
        self.max_exponent: tf.Tensor = tf.cast(max_exponent, tf.float32)
        self.trainable: bool = trainable

    def build(self, input_shape) -> None:
        """
        This method initializes the weights of the layer.

        Args:
            input_shape: The shape of the input to the layer, used to shape the weights.
        """
        self.exponent = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.initial_exponent),
            trainable=self.trainable,
            name='exponent',
        )
        self.bias = self.add_weight(
            shape=(),
            initializer='zeros',
            trainable=self.trainable,
            name='bias',
        )
        super(PowerSplineActivation, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies the power spline activation function to the inputs.

        Args:
            inputs (tf.Tensor): Input tensor to be transformed.

        Returns:
            tf.Tensor: Transformed tensor with the same shape as input.
        """
        clipped_exponent: tf.Tensor = tf.clip_by_value(self.exponent, -self.max_exponent, self.max_exponent)
        x_safe: tf.Tensor = tf.clip_by_value(inputs + self.bias, self.epsilon, 1.)
        return tf.pow(x_safe, clipped_exponent)

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A Python dictionary containing the layer configuration, ensuring that all 
            initialization parameters are included for correct layer reconstruction upon loading.
        """
        config = super(PowerSplineActivation, self).get_config()
        config.update({
            'initial_exponent': self.initial_exponent.numpy(),
            'epsilon': self.epsilon,
            'max_exponent': self.max_exponent.numpy(),
            'trainable': self.trainable,
        })
        return config


@tf.keras.utils.register_keras_serializable(package='tkan', name='LinspaceInitializer')
class LinspaceInitializer(tf.keras.initializers.Initializer):
    """
    A custom TensorFlow initializer that generates values linearly spaced between a specified start and stop value.
    """
    def __init__(self, start: float, stop: float, num: int):
        """
        Initialize the LinspaceInitializer with start, stop, and num values.

        Args:
            start (float): The starting value of the sequence.
            stop (float): The ending value of the sequence.
            num (int): The number of values to generate.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.num = num

    def __call__(self, shape, dtype=None):
        """
        Generates a tensor with the specified shape, filled with linearly spaced values.

        Args:
            shape (tuple): The shape of the tensor to be generated. Note that the number of elements in the tensor
                           must match the `num` parameter provided during initialization.
            dtype (optional): The data type of the tensor to be returned.

        Returns:
            tf.Tensor: A tensor with linearly spaced values from start to stop.
        """
        result = tf.linspace(self.start, self.stop, self.num)
        if dtype is not None:
            result = tf.cast(result, dtype)
        return result

    def get_config(self):
        """
        Returns the configuration of the initializer.

        Returns:
            dict: Configuration dictionary.
        """
        return {'start': self.start, 'stop': self.stop, 'num': self.num}


@tf.keras.utils.register_keras_serializable(package='tkan', name='BSplineActivation')
class BSplineActivation(Layer):
    """
    A custom Keras Layer implementing a B-Spline activation function with trainable coefficients,
    coupled with a SiLU activation for hybrid functionality.
    """
    def __init__(self, num_bases: int = 10, order: int = 3, **kwargs):
        """
        Initializes the BSplineActivation layer with the number of bases and the order of the spline.

        Args:
            num_bases (int): The number of basis functions in the B-Spline.
            order (int): The order of the spline.
        """
        super(BSplineActivation, self).__init__(**kwargs)
        self.num_bases = num_bases
        self.order = order
        self.w = self.add_weight(name='w', shape=(), initializer='glorot_uniform', trainable=True)

    def build(self, input_shape):
        """
        Build the weights of the layer.

        Args:
            input_shape: The shape of the input to the layer.
        """
        self.coefficients = self.add_weight(
            shape=(self.num_bases,),
            initializer='zeros',
            trainable=True,
            name='coefficients'
        )
        self.bases = self.add_weight(
            name="bases",
            shape=(self.num_bases,),
            initializer=LinspaceInitializer(0.0, 1.0, self.num_bases),
            trainable=False
        )
        super(BSplineActivation, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the hybrid B-Spline and SiLU activation function to the inputs.

        Args:
            inputs (tf.Tensor): Input tensor to be transformed.

        Returns:
            tf.Tensor: The transformed tensor, combining both SiLU and B-Spline outputs.
        """
        x = inputs
        silu = x * tf.sigmoid(x)  # SiLU activation
        spline_output = self.compute_spline(x)
        return self.w * (silu + spline_output)

    def compute_spline(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the B-Spline values for the given inputs.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: Calculated spline values.
        """
        safe_x = tf.clip_by_value(x, K.epsilon(), 1.0)
        expanded_x = tf.expand_dims(safe_x, -1)
        basis_function_values = tf.pow(expanded_x - self.bases, self.order)
        spline_values = tf.reduce_sum(self.coefficients * basis_function_values, axis=-1)
        return spline_values

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary containing settings for number of bases and spline order.
        """
        config = super(BSplineActivation, self).get_config()
        config.update({"num_bases": self.num_bases, "order": self.order})
        return config
