import time
import keras
from keras import ops
from keras import random
from keras import activations
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import InputSpec, Layer, RNN, Dense

# Assume KANLinear is defined elsewhere in a backend-agnostic manner
from keras_efficient_kan import KANLinear

def get_backend():
    import os
    return os.environ.get('KERAS_BACKEND', 'tensorflow')

@keras.utils.register_keras_serializable(package="tkan", name="TKANCell")
class TKANCell(Layer):
    """Cell class for the TKAN layer.
    Modification of the LSTM implementation in keras 3 in order to provide fully seamless integration with TF, torch and jax backend

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
        sub_kan_configs=None,
        sub_kan_output_dim=None,
        sub_kan_input_dim=None,
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
        super().__init__(**kwargs)
        self.units = units
        self.sub_kan_configs = sub_kan_configs or [None]
        self.sub_kan_output_dim = sub_kan_output_dim
        self.sub_kan_input_dim = sub_kan_input_dim
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self._dropout_mask = None
        self.dropout = min(1.0, max(0.0, dropout))
        self._recurrent_dropout_mask = None
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed if seed is not None else int(time.time())
        self.state_size = [units, units] + [1 for _ in self.sub_kan_configs]
        self.output_size = units
        self.backend = get_backend()
    

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        if self.sub_kan_input_dim is None:
            self.sub_kan_input_dim = input_dim
        if self.sub_kan_output_dim is None:
            self.sub_kan_output_dim = input_dim
    
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
    
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return ops.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get("ones")((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 3,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
    
        self.tkan_sub_layers = []
        for config in self.sub_kan_configs:
            if config is None:
                layer = KANLinear(self.sub_kan_output_dim, use_layernorm=True)
            elif isinstance(config, (int, float)):
                layer = KANLinear(self.sub_kan_output_dim, spline_order=config, use_layernorm=True)
            elif isinstance(config, dict):
                layer = KANLinear(self.sub_kan_output_dim, **config, use_layernorm=True)
            else:
                layer = Dense(self.sub_kan_output_dim, activation=config)
            layer.build((input_shape[0], self.sub_kan_input_dim))
            self.tkan_sub_layers.append(layer)
    
        self.sub_tkan_kernel = self.add_weight(
            shape=(len(self.tkan_sub_layers), self.sub_kan_output_dim * 2),
            name="sub_tkan_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.sub_tkan_recurrent_kernel_inputs = self.add_weight(
            shape=(len(self.tkan_sub_layers), input_shape[-1], self.sub_kan_input_dim),
            name="sub_tkan_recurrent_kernel_inputs",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.sub_tkan_recurrent_kernel_states = self.add_weight(
            shape=(len(self.tkan_sub_layers), self.sub_kan_output_dim, self.sub_kan_input_dim),
            name="sub_tkan_recurrent_kernel_states",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.aggregated_weight = self.add_weight(
            shape=(len(self.tkan_sub_layers) * self.sub_kan_output_dim, self.units),
            name="aggregated_weight",
            initializer="glorot_uniform",
        )
        self.aggregated_bias = self.add_weight(
            shape=(self.units,),
            name="aggregated_bias",
            initializer="zeros",
        )
    
        self.built = True

    def _generate_dropout_mask(self, inputs):
        if 0 < self.dropout < 1:
            seed_generator = random.SeedGenerator(self.seed)
            return random.dropout(
                ops.ones_like(inputs),
                self.dropout,
                seed=seed_generator
            )
        return None

    def _generate_recurrent_dropout_mask(self, states):
        if 0 < self.recurrent_dropout < 1:
            seed_generator = random.SeedGenerator(self.seed + 1)
            return random.dropout(
                ops.ones_like(states),
                self.recurrent_dropout,
                seed=seed_generator
            )
        return None

    def call(self, inputs, states, training=None):
        if self.backend == 'tensorflow':
            return self._call_tensorflow(inputs, states, training)
        else:
            return self._call_generic(inputs, states, training)

    def _call_tensorflow(self, inputs, states, training=False):
        """
        Needed to be different in order to be jit_compile friendly
        """
        import tensorflow as tf
        h_tm1 = states[0]  # Previous memory state for the LSTM part.
        c_tm1 = states[1]  # Previous carry state for the LSTM part.
        sub_states = states[2:]  # Previous states for each sub-layer.
    
        batch_size = tf.shape(inputs)[0]
        if training:
            self.seed = (self.seed + 1) % (2**32 - 1) 
            if self.dropout > 0.0:
                inputs = inputs * self._generate_dropout_mask(inputs)
            if self.recurrent_dropout > 0.0:
                h_tm1 = h_tm1 * self._generate_recurrent_dropout_mask(h_tm1)
    
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
            sub_kernel_x, sub_kernel_h = self.sub_tkan_recurrent_kernel_inputs[idx], self.sub_tkan_recurrent_kernel_states[idx]
            agg_input = inputs @ sub_kernel_x + sub_state @ sub_kernel_h
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = tf.split(self.sub_tkan_kernel[idx], 2, axis=0)
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

    def _call_generic(self, inputs, states, training=None):
        h_tm1 = states[0]  # Previous memory state
        c_tm1 = states[1]  # Previous carry state
        sub_states = states[2:]  # Previous states for sub-layers

        if training:
            self.seed = (self.seed + 1) % (2**32 - 1) 
            dp_mask = self._generate_dropout_mask(inputs)
            rec_dp_mask = self._generate_recurrent_dropout_mask(h_tm1)
            if dp_mask is not None:
                inputs *= dp_mask
            if rec_dp_mask is not None:
                h_tm1 *= rec_dp_mask

        if self.use_bias:
            gates = ops.matmul(inputs, self.kernel) + ops.matmul(h_tm1, self.recurrent_kernel) + self.bias
        else:
            gates = ops.matmul(inputs, self.kernel) + ops.matmul(h_tm1, self.recurrent_kernel)
        
        i, f, c = ops.split(self.recurrent_activation(gates), 3, axis=-1)

        c = f * c_tm1 + i * self.activation(c)

        sub_outputs = []
        new_sub_states = []

        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x, sub_kernel_h = self.sub_tkan_recurrent_kernel_inputs[idx], self.sub_tkan_recurrent_kernel_states[idx]
            agg_input = inputs @ sub_kernel_x + sub_state @ sub_kernel_h
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = ops.split(self.sub_tkan_kernel[idx], 2, axis=0)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        aggregated_sub_output = ops.concatenate(sub_outputs, axis=-1)
        aggregated_input = ops.dot(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias

        o = self.recurrent_activation(aggregated_input)

        h = o * self.activation(c)

        return h, [h, c] + new_sub_states
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_kan_configs": self.sub_kan_configs,
            "sub_kan_output_dim": self.sub_kan_output_dim,
            "sub_kan_input_dim": self.sub_kan_input_dim,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        })
        return config

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or self.compute_dtype
        return [
            ops.zeros((batch_size, self.units), dtype=dtype),
            ops.zeros((batch_size, self.units), dtype=dtype)
        ] + [ops.zeros((batch_size, self.sub_kan_output_dim), dtype=dtype) for _ in range(len(self.tkan_sub_layers))]


@keras.utils.register_keras_serializable(package="tkan", name="TKAN")
class TKAN(RNN):
    def __init__(
        self,
        units,
        sub_kan_configs=None,
        sub_kan_output_dim=None,
        sub_kan_input_dim=None,
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
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        seed=None,
        **kwargs,
    ):
        cell = TKANCell(
            units,
            sub_kan_configs=sub_kan_configs,
            sub_kan_output_dim=sub_kan_output_dim,
            sub_kan_input_dim=sub_kan_input_dim,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)
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
    def sub_kan_configs(self):
        return self.cell.sub_kan_configs

    @property
    def sub_kan_output_dim(self):
        return self.cell.sub_kan_output_dim

    @property
    def sub_kan_input_dim(self):
        return self.cell.sub_kan_input_dim
    
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


    def build(self, input_shape):
        super().build(input_shape)
    
    def get_config(self):
        config = {
            "units": self.units,
            "sub_kan_configs": self.sub_kan_configs,
            "sub_kan_output_dim": self.cell.sub_kan_output_dim,
            "sub_kan_input_dim": self.cell.sub_kan_input_dim,
            "activation": activations.serialize(self.cell.activation),
            "recurrent_activation": activations.serialize(self.cell.recurrent_activation),
            "use_bias": self.cell.use_bias,
            "kernel_initializer": initializers.serialize(self.cell.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.cell.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.cell.bias_initializer),
            "unit_forget_bias": self.cell.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.cell.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.cell.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.cell.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.cell.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.cell.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.cell.bias_constraint),
            "dropout": self.cell.dropout,
            "recurrent_dropout": self.cell.recurrent_dropout,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)
