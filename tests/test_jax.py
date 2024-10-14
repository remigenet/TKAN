import os
import tempfile
BACKEND = 'jax'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random
from keras.models import Model, load_model
from keras.layers import Input
from tkan import TKAN

def generate_random_tensor(shape):
    return random.normal(shape=shape, dtype=backend.floatx())

def test_tkan_sequence():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    
    tkan_layer = TKAN(10, return_sequences=False)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_sequence = tkan_layer(input_sequence)
    assert output_sequence.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, but got {output_sequence.shape}"

    tkan_layer = TKAN(10, return_sequences=True)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_sequence = tkan_layer(input_sequence)
    assert output_sequence.shape == (batch_size, time_steps, 10), f"Expected shape {(batch_size, time_steps, 10)}, but got {output_sequence.shape}"

def test_tkan_stateful():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 2, 10, 3
    
    # Use the functional API
    inputs = keras.Input(batch_shape=(batch_size, time_steps, features))
    tkan_layer = TKAN(4, stateful=True, return_sequences=True)
    outputs = tkan_layer(inputs)
    model = keras.Model(inputs, outputs)
    
    model.compile('rmsprop', 'mse')
    
    out1 = model.predict(generate_random_tensor((batch_size, time_steps, features)))
    out2 = model.predict(generate_random_tensor((batch_size, time_steps, features)))
    assert ops.any(ops.not_equal(out1, out2)), "Stateful TKAN should produce different outputs for different inputs"
    
    # Reset the states of the TKAN layer
    tkan_layer.reset_states()
    
    out3 = model.predict(generate_random_tensor((batch_size, time_steps, features)))
    assert ops.any(ops.not_equal(out1, out3)), "TKAN should produce different outputs after reset_states"

def test_tkan_return_sequences_and_states():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 6, 10, 3
    inputs = keras.Input((time_steps, features))
    layer = TKAN(4, sub_kan_configs=[None, 3], return_sequences=True, return_state=True)
    outputs = layer(inputs)
    assert len(outputs) == 5, "TKAN with return_state and 2 sublayer should return 5 outputs"
    
    model = keras.Model(inputs, outputs)
    model.compile('rmsprop', 'mse')
    
    x = generate_random_tensor((batch_size, time_steps, features))
    y = [
        generate_random_tensor((batch_size, time_steps, 4)),  # output sequence
        generate_random_tensor((batch_size, 4)),  # final memory state
        generate_random_tensor((batch_size, 4))   # final carry state
    ]
    
    model.fit(x, y, epochs=1, batch_size=2)

def test_tkan_masking():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 6, 10, 3
    inputs = keras.Input((time_steps, features))
    masked = keras.layers.Masking(mask_value=0.)(inputs)
    layer = TKAN(4)
    outputs = layer(masked)

    model = keras.Model(inputs, outputs)
    model.compile('rmsprop', 'mse')

    # Create input with some zero vectors to be masked
    x = generate_random_tensor((batch_size, time_steps, features))
    if keras.backend.backend() == 'jax':
        x = x.at[:, [0, 5]].set(0)  # Set first and sixth time steps to zero
    elif keras.backend.backend() == 'torch':
        x = x.cpu().numpy()  # Move to CPU before converting to NumPy
        x[:, [0, 5]] = 0  # Set first and sixth time steps to zero
        x = keras.ops.convert_to_tensor(x)  # Convert back to tensor
    else:
        x = x.numpy()
        x[:, [0, 5]] = 0  # Set first and sixth time steps to zero
        x = keras.ops.convert_to_tensor(x)  # Convert back to tensor

    y = generate_random_tensor((batch_size, 4))
    model.fit(x, y, epochs=1, batch_size=2)
    
def test_tkan_dropout():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 6, 10, 3
    layer = TKAN(4, dropout=0.2, recurrent_dropout=0.2)
    inputs = generate_random_tensor((batch_size, time_steps, features))
    
    # Run twice to check if dropout is applied
    output1 = layer(inputs, training=True)
    output2 = layer(inputs, training=True)
    
    assert ops.any(ops.not_equal(output1, output2)), "Dropout should cause different outputs in training mode"
    
    # Check if dropout is not applied in inference mode
    output3 = layer(inputs, training=False)
    output4 = layer(inputs, training=False)
    
    assert ops.all(ops.equal(output3, output4)), "Outputs should be the same in inference mode"

def test_tkan_save_and_load():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    units = 16

    # Create and compile the model
    inputs = Input(shape=(time_steps, features))
    tkan_layer = TKAN(units, return_sequences=True)
    outputs = tkan_layer(inputs)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    # Generate some random data
    x_train = generate_random_tensor((batch_size, time_steps, features))
    y_train = generate_random_tensor((batch_size, time_steps, units))

    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)

    # Get predictions before saving
    predictions_before = model.predict(x_train, verbose=False)

    # Save the model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'tkan_model.keras')
        model.save(model_path)

        # Load the model
        loaded_model = load_model(model_path)

    # Get predictions after loading
    predictions_after = loaded_model.predict(x_train, verbose=False)

    # Compare predictions
    assert ops.all(ops.equal(predictions_before, predictions_after)), "Predictions should be the same after loading"

    # Test that the loaded model can be used for further training
    loaded_model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)

    print("TKAN model successfully saved, loaded, and reused.")
