from typing import List, Union, Optional, Callable, Tuple
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tkan import BSplineActivation, PowerSplineActivation, FixedSplineActivation


@tf.keras.utils.register_keras_serializable(package='tkan', name='TKAN')
class TKAN(Layer):
    """
    TKAN (Temporal Knowledge Activation Network) is a custom TensorFlow layer that integrates multiple
    activation functions into a single network architecture. It combines features extracted using different
    activation mechanisms in a recurrent neural network (RNN) style framework to enhance feature learning.
    The layer allows the use of non-standard activation functions like B-spline, Power Spline, and Fixed Spline
    in addition to traditional activations. The architecture is designed for sequence processing tasks, capable
    of returning either the last output or the full sequence of outputs.

    Attributes:
        activation_funcs (List[Union[str, int, None, Callable]]): List specifying the activation function for each
            sub-layer. It supports standard activation names, custom activation functions, or special spline-based
            activations identified by numeric initial exponents.
        num_outputs (int): The number of output neurons in the layer.
        return_sequences (bool): Flag to determine whether to return the full sequence of outputs or just the final output.
        trainable_power_spline (bool): If set to True, the parameters of the PowerSplineActivation functions can be trained.
    """

    def __init__(self, 
                 activation_funcs: List[Union[str, int, None, Callable]], 
                 num_outputs: int, 
                 return_sequences: bool = False, 
                 trainable_power_spline: bool = False, 
                 **kwargs) -> None:
        """
        Initializes a TKAN layer with specified activations, number of output units, and other configurations.

        Args:
            activation_funcs (List[Union[str, int, None, Callable]]): A list containing elements that specify the
                type of activation function for each sub-layer. This can be a string for TensorFlow activations,
                an integer or float for spline activations (indicating the initial exponent), None for a default
                B-Spline activation, or a callable object for custom activations.
            num_outputs (int): Specifies the number of output units.
            return_sequences (bool, optional): Determines if the layer should return the sequence of outputs
                (True) or just the final output (False). Defaults to False.
            trainable_power_spline (bool, optional): Allows the PowerSplineActivation parameters to be trainable if True.
                Defaults to False.
            **kwargs: Standard parameters for Keras layers (e.g., `name`).
        """
        super(TKAN, self).__init__(**kwargs)
        self.activation_funcs = activation_funcs
        self.num_outputs = num_outputs
        self.return_sequences = return_sequences
        self.aggregation_transform = tf.keras.layers.Dense(num_outputs, activation="sigmoid")
        # Sub-layer LSTM components
        self.global_biases = {}
        self.global_lstm_gates = {}
        self.global_recurrent_weights = {}
        self.sub_layers_lstm_weights = {}
        self.sub_layers = []
        for act in self.activation_funcs:
            if act is None:
                self.sub_layers.append(tf.keras.layers.Dense(1, activation=BSplineActivation()))
            elif isinstance(act, (int, float)):
                if trainable_power_spline:
                    self.sub_layers.append(tf.keras.layers.Dense(1, activation=PowerSplineActivation(initial_exponent=act)))
                else:
                    self.sub_layers.append(tf.keras.layers.Dense(1, activation=FixedSplineActivation(exponent=act)))
            else:
                self.sub_layers.append(tf.keras.layers.Dense(1, activation=act))


    def build(self, input_shape: Tuple[int]):
        """
        Creates the weights of the layer. This method initializes the weights for global LSTM gates,
        recurrent weights, biases, and sub-layer specific LSTM-like weights. It is called automatically
        the first time the layer is used.

        Args:
            input_shape (Tuple[int]): The shape of the input to the layer, typically (batch_size, sequence_length, num_features).
        """
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        num_features = input_shape[2]
        # Global LSTM gate weights
        self.global_lstm_gates = {
            'Wi': self.add_weight(shape=(num_features, self.num_outputs), initializer='glorot_uniform', trainable=True),
            'Wf': self.add_weight(shape=(num_features, self.num_outputs), initializer='glorot_uniform', trainable=True),
            'Wc': self.add_weight(shape=(num_features, self.num_outputs), initializer='glorot_uniform', trainable=True)
        }

        self.global_recurrent_weights = {
            'Ui': self.add_weight(shape=(self.num_outputs, self.num_outputs), initializer='orthogonal', trainable=True),
            'Uf': self.add_weight(shape=(self.num_outputs, self.num_outputs), initializer='orthogonal', trainable=True),
            'Uc': self.add_weight(shape=(self.num_outputs, self.num_outputs), initializer='orthogonal', trainable=True)
        }

        self.global_biases = {
            'bi': self.add_weight(shape=(self.num_outputs,), initializer='zeros', trainable=True),
            'bf': self.add_weight(shape=(self.num_outputs,), initializer='ones', trainable=True),
            'bc': self.add_weight(shape=(self.num_outputs,), initializer='zeros', trainable=True)
        }
        self.sub_layers_lstm_weights = {
            'Whx': self.add_weight(shape=(len(self.sub_layers), num_features), initializer='orthogonal', trainable=True),
            'Whh': self.add_weight(shape=(len(self.sub_layers), num_features), initializer='orthogonal', trainable=True),
            'U3': self.add_weight(shape=(len(self.sub_layers), 1), initializer='orthogonal', trainable=True),
            'U4': self.add_weight(shape=(len(self.sub_layers), 1), initializer='orthogonal', trainable=True)
        }
        for layer in self.sub_layers:
            layer.build((batch_size, num_features))
        super(TKAN, self).build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Logic for the layer's forward pass. Handles the computation of LSTM gate activations,
        sub-layer processing, and the sequential accumulation of outputs.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).
            training (bool, optional): Whether the call is for training or inference. This parameter
                influences behavior like dropout.

        Returns:
            tf.Tensor: The output tensor. If return_sequences is True, returns all time steps (batch_size, sequence_length, num_outputs).
                       Otherwise, returns only the last time step's output (batch_size, num_outputs).
        """
        batch_size = tf.shape(inputs)[0]
        sub_hidden_states = [tf.zeros((batch_size, 1)) for _ in self.activation_funcs]
        sub_cell_states = [tf.zeros((batch_size, 1)) for _ in self.activation_funcs]
        global_hidden_state = tf.zeros((batch_size, self.num_outputs))
        global_cell_state = tf.zeros((batch_size, self.num_outputs))
        outputs = []

        for t in range(inputs.shape[1]):
            current_input = inputs[:, t, :]
            new_sub_hidden_states = []
            new_sub_cell_states = []
            sub_outputs = []

            global_xf = tf.nn.sigmoid(tf.linalg.matmul(current_input, self.global_lstm_gates['Wf']) + tf.linalg.matmul(global_hidden_state , self.global_recurrent_weights['Uf']) + self.global_biases['bf'])
            global_xi = tf.nn.sigmoid(tf.linalg.matmul(current_input, self.global_lstm_gates['Wi']) + tf.linalg.matmul(global_hidden_state , self.global_recurrent_weights['Ui']) + self.global_biases['bi'])


            for idx, sub_layer in enumerate(self.sub_layers):
                agg_input = current_input * self.sub_layers_lstm_weights['Whx'][idx] + sub_hidden_states[idx] * self.sub_layers_lstm_weights['Whh'][idx]
                sub_output = sub_layer(agg_input)
                sub_hidden_states[idx] = self.sub_layers_lstm_weights['U3'][idx] * sub_output + sub_hidden_states[idx] * self.sub_layers_lstm_weights['U4'][idx]
                sub_outputs.append(sub_output)
            # Global LSTM processing of aggregated sub-outputs
            aggregated_input = tf.concat(sub_outputs, axis=-1)
            # Transform aggregated input to match LSTM gate dimensions
            global_xo = self.aggregation_transform(aggregated_input)

            c_tilde = tf.nn.sigmoid(tf.linalg.matmul(current_input, self.global_lstm_gates['Wc']) + tf.linalg.matmul(global_hidden_state , self.global_recurrent_weights['Uc']) + self.global_biases['bc'])

            global_cell_state = global_xf * global_cell_state + global_xi * c_tilde
            global_hidden_state = global_xo * tf.nn.tanh(global_cell_state)
            
            if self.return_sequences:
                outputs.append(global_hidden_state)

        return tf.stack(outputs, axis=1) if self.return_sequences else global_hidden_state


    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        """
        Computes the output shape of the layer based on the given input shape.

        Args:
            input_shape (Tuple[int, int, int]): Tuple representing the shape of the input.
                The tuple is expected to contain three integers:
                (batch_size, sequence_length, num_features).

        Returns:
            Union[Tuple[int, int, int], Tuple[int, int]]: Depending on the value of `return_sequences`,
            returns a tuple indicating the shape of the output tensor. If `return_sequences` is True,
            the output shape will be (batch_size, sequence_length, num_outputs), reflecting the full sequence
            of outputs. If False, the output shape will be (batch_size, num_outputs), reflecting the final
            output state only.
        """
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.num_outputs)
        else:
            return (input_shape[0], self.num_outputs)


