# TKAN: Temporal Kolmogorov-Arnold Networks

TKAN (Temporal Kolmogorov-Arnold Networks) is a neural network architecture designed to enhance multi-horizon time series forecasting. This TensorFlow implementation integrates TKAN as a layer within sequential models, facilitating the use of advanced neural network techniques in practical applications. It is the original implementation of the [paper](https://arxiv.org/abs/2405.07344)

![TKAN representation](image/TKAN.drawio.png)

## Installation

Install TKAN directly from PyPI:

```bash
pip install tkan
```

Dependencies are managed using pyproject.toml.

## Usage

TKAN can be used within TensorFlow models to handle complex sequential patterns in data.
It's implementation reproduce architecture of RNN in tensorflow with Cell class and Layer that inherits from RNN in order to provide a perfect integrations with tensorflow.
Here is an example that demonstrates how to use TKAN with B-spline activations in a sequential model:

```python
from temporal_kan import TKAN
import tensorflow as tf

# Example model using TKAN with B-spline activations
model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=X_train_seq.shape[1:]),
      TKAN(100, tkan_activations=[{'spline_order': 3, 'grid_size': 10}, {'spline_order': 1, 'grid_size': 5}, {'spline_order': 4, 'grid_size': 6}, ], return_sequences=True, use_bias=True), #Define the params of the KANLinear as dict as here
      TKAN(100, tkan_activations=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True), #Use float or int to specify only the exponent of the spline
      TKAN(100, tkan_activations=['relu', 'relu', 'relu', 'relu', 'relu'], return_sequences=True, use_bias=True), #Or use string to specify the standard tensorflow activation using Dense in sublayers instead of KANLinear
      TKAN(100, tkan_activations=[None for _ in range(3)], return_sequences=False, use_bias=True), # Or put None for default activation
      tf.keras.layers.Dense(y_train_seq.shape[1]),
])
```

You can find a more complete example with comparison with LSTM in the example folder.

### Activation Function Flexibility

TKAN layers are highly flexible with regards to activation functions. They can be configured using various types of activations:
- *Callable classes*: Custom classes like BSplineActivation allow for sophisticated configurations.
- *Integers or floats*: Specify an initial exponent for a simple power spline activation.
- *None*: Defaults to BSplineActivation with an order of 3.
- *Strings*: Utilizes standard TensorFlow activation functions.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
