# TKAN: Temporal Kolmogorov-Arnold Networks

TKAN (Temporal Kolmogorov-Arnold Networks) is a neural network architecture designed to enhance multi-horizon time series forecasting. This TensorFlow implementation integrates TKAN as a layer within sequential models, facilitating the use of advanced neural network techniques in practical applications. It is the original implementation of the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824002)

## Installation

Install TKAN directly from PyPI:

```bash
pip install tkan
```

Dependencies are managed using pyproject.toml.

## Usage

TKAN can be used within TensorFlow models to handle complex sequential patterns in data. Here is an example that demonstrates how to use TKAN with B-spline activations in a sequential model:

```python
from temporal_kan import TKAN, BSplineActivation
import tensorflow as tf

# Example model using TKAN with B-spline activations
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]),
    TKAN(activation_funcs=[BSplineActivation(i) for i in range(5)], num_outputs=100, return_sequences=True),
    TKAN(activation_funcs=[BSplineActivation(i) for i in range(1, 4)], num_outputs=100, return_sequences=False),
    tf.keras.layers.Dense(y_train.shape[1], activation='linear')
], name=f'Sequential_Bspline')
```

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
