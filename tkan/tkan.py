import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import InputSpec, Layer, RNN
from tkan import KANLinear


class DropoutRNNCell:
    """Direct copy of the keras class (https://github.com/keras-team/keras/blob/v3.3.3/keras/src/layers/rnn/dropout_rnn_cell.py)
    Object that holds dropout-related functionality for RNN cells.

    This class is not a standalone RNN cell. It suppose to be used with a RNN
    cell by multiple inheritance. Any cell that mix with class should have
    following fields:

    - `dropout`: a float number in the range `[0, 1]`.
        Dropout rate for the input tensor.
    - `recurrent_dropout`: a float number in the range `[0, 1]`.
        Dropout rate for the recurrent connections.
    - `seed_generator`, an instance of `backend.random.SeedGenerator`.

    This object will create and cache dropout masks, and reuse them for
    all incoming steps, so that the same mask is used for every step.
    """

    def get_dropout_mask(self, step_input):
        if not hasattr(self, "_dropout_mask"):
            self._dropout_mask = None
        if self._dropout_mask is None and self.dropout > 0:
            ones = tf.ones_like(step_input)
            self._dropout_mask = K.dropout(
                ones, level=self.dropout, seed=self.seed_generator
            )
        return self._dropout_mask

    def get_recurrent_dropout_mask(self, step_input):
        if not hasattr(self, "_recurrent_dropout_mask"):
            self._recurrent_dropout_mask = None
        if self._recurrent_dropout_mask is None and self.recurrent_dropout > 0:
            ones = tf.ones_like(step_input)
            self._recurrent_dropout_mask = K.dropout(
                ones, level=self.recurrent_dropout, seed=self.seed_generator
            )
        return self._recurrent_dropout_mask

    def reset_dropout_mask(self):
        """Reset the cached dropout mask if any.

        The RNN layer invokes this in the `call()` method
        so that the cached mask is cleared after calling `cell.call()`. The
        mask should be cached across all timestep within the same batch, but
        shouldn't be cached between batches.
        """
        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self):
        self._recurrent_dropout_mask = None


@tf.keras.utils.register_keras_serializable(package="tkan", name="TKANCell")
class TKANCell(Layer, DropoutRNNCell):
    """Cell class for the TKAN layer.
    Modification of the LSTM implementation in tensorflow in order to provide fully seamless integration within the framework

    This class processes one step within the whole time sequence input, whereas
    `TKAN` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.TKANCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.TKANCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)
    """
    def __init__(
        self,
        units,
        tkan_activations=None,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        tkan_activations = tkan_activations or [None]
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.sub_tkan_initializer = initializers.get(recurrent_initializer)
        self.recurrent_sub_tkan_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed


        if seed is not None:
            self.seed_generator = tf.random.experimental.Generator.from_seed(seed=seed)
        else:
            try:
                self.seed_generator = tf.random.experimental.Generator.from_non_deterministic_state()
            except RuntimeError:
                # Fallback to a deterministic generator with a fixed seed
                self.seed_generator = tf.random.experimental.Generator.from_seed(seed=0)

        self.unit_forget_bias = unit_forget_bias
        self.state_size = [units, units] + [1 for _ in tkan_activations]
        self.output_size = units

        self.tkan_sub_layers = []
        
        for act in tkan_activations:
            if act is None:
                self.tkan_sub_layers.append(KANLinear(1))
            elif isinstance(act, (int, float)):
                self.tkan_sub_layers.append(KANLinear(1, spline_order=act))
            elif isinstance(act, dict):
                self.tkan_sub_layers.append(KANLinear(1, **act))
            else:
                self.tkan_sub_layers.append(tf.keras.layers.Dense(1, activation=act))


    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        name = self.name
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name=f"{name}_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name=f"{name}_recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.sub_tkan_kernel = self.add_weight(
            shape=(len(self.tkan_sub_layers), 2),
            name=f"{name}_sub_tkan_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.sub_tkan_recurrent_kernel = self.add_weight(
            shape=(len(self.tkan_sub_layers), input_shape[1] * 2),
            name=f"{name}_sub_tkan_recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.aggregated_weight = self.add_weight(
            shape=(len(self.tkan_sub_layers), self.units),
            initializer='glorot_uniform',
            name=f'{name}_aggregated_weight'
        )
        self.aggregated_bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name=f'{name}_aggregated_bias'
        )
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return tf.concat(
                        [
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.get("ones")((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units,), *args, **kwargs),
                        ],
                        axis=0
                    )
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 3,),
                name=f"{name}_bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        for layer in self.tkan_sub_layers:
            layer.build((input_shape[0], input_shape[1]))

        self.built = True

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]  # Previous memory state for the LSTM part.
        c_tm1 = states[1]  # Previous carry state for the LSTM part.
        sub_states = states[2:]  # Previous states for each sub-layer.

        batch_size = tf.shape(inputs)[0]
        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if training and self.dropout > 0.0:
            inputs = inputs * dp_mask
        if training and self.recurrent_dropout > 0.0:
            h_tm1 = h_tm1 * rec_dp_mask

        # Preallocate tensors for sub-layer outputs and new states
        sub_outputs = tf.TensorArray(dtype=tf.float32, size=len(self.tkan_sub_layers))
        new_sub_states = tf.TensorArray(dtype=tf.float32, size=len(self.tkan_sub_layers))

        # Split the kernel and compute input projections for gates
        if self.use_bias:
          x_i, x_f, x_c = tf.split(self.recurrent_activation(tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel) + self.bias), 3, axis=1)
        else:
          x_i, x_f, x_c = tf.split(self.recurrent_activation(tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel)), 3, axis=1)

        # Process each sub-layer
        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_h, sub_kernel_x = tf.split(self.sub_tkan_recurrent_kernel[idx, :], 2, axis=0)
            agg_input = inputs * sub_kernel_x + sub_state * sub_kernel_h
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = tf.split(self.sub_tkan_kernel[idx, :], 2, axis=0)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x

            sub_outputs = sub_outputs.write(idx, sub_output)
            new_sub_states = new_sub_states.write(idx, new_sub_state)

        # Stack outputs and states for further processing
        sub_outputs = sub_outputs.stack()

        # Aggregate sub-layer outputs using weights and biases
        aggregated_sub_output = tf.reshape(sub_outputs, (batch_size, -1))
        aggregated_input = tf.matmul(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias

        xo = self.recurrent_activation(aggregated_input)

        c = x_f * c_tm1 + x_i * x_c 

        # # Compute the TKAN cell's new states
        h = xo * self.activation(c)

        # Prepare output and new states
        new_states = [h, c] + tf.unstack(new_sub_states.stack())
        return h, new_states


    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "tkan_sub_layers": self.tkan_sub_layers,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or self.compute_dtype
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype)
        ] + [tf.zeros((batch_size, 1), dtype=dtype) for _ in range(len(self.tkan_sub_layers))]




@tf.keras.utils.register_keras_serializable(package="tkan", name="TKAN")
class TKAN(RNN):
    """Temporal Kolmogorow-Arnold Network - Inzirillo & Genet 2024.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> tkan = TKAN(4)
    >>> output = tkan(inputs)
    >>> output.shape
    (32, 4)
    >>> tkan = TKAN(
    ...     4, return_sequences=True, return_state=True)
    >>> whole_seq_output, states = tkan(inputs)
    >>> final_memory_state, final_carry_state, sub_tkan_layers_states = states[0], states[1], states[2:]
    >>> whole_seq_output.shape
    (32, 10, 4)
    >>> final_memory_state.shape
    (32, 4)
    >>> final_carry_state.shape
    (32, 4)
    >>> final_carry_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        use_cudnn: Ignored, no implementation associated for the moment

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.
    """

    def __init__(
        self,
        units,
        tkan_activations=None,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        use_cudnn="auto",
        **kwargs,
    ):
        cell = TKANCell(
            units,
            tkan_activations=tkan_activations,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            unit_forget_bias=unit_forget_bias,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get("dtype", None),
            trainable=kwargs.get("trainable", True),
            name="tkan_cell",
            seed=seed,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = [InputSpec(ndim=3)]
        

    def inner_loop(self, sequences, initial_state, mask, training=False):
        if isinstance(mask, (list, tuple)):
            mask = mask[0]
        return super().inner_loop(
            sequences, initial_state, mask=mask, training=training
        )

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences, mask=mask, training=training, initial_state=initial_state
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            "units": self.units,
            "tkan_activations": [activations.serialize(lay.activation) for lay in self.cell.tkan_sub_layers],
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
