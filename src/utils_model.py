import tensorflow as tf


def build_input_layer(input_size: tuple[int, int]):
    """
    Build a keras input layer
    :param input_size: The shape of the input
    :return: A keras input layer
    """
    inputs = tf.keras.layers.Input(shape=input_size)
    return inputs


def build_dense_block(inputs, neurons, output_neuron, output_activation, dropout_rate, should_flatten=False):
    """
    Add a dense block to the layers in inputs
    :param inputs: The model in which we will add a dense block
    :param neurons: a list of neurons' numbers per hidden dense layer in the model
    :param output_neuron: the number of neuron in the output dense layer in the model
    :param output_activation:
    :param dropout_rate: the dropout rate
    :param should_flatten: parameter to decide if we add a flatten layer before the dense layer block
    :return: the model with a dense block add to it
    """
    # Checking function parameter
    if type(neurons) is not list:
        raise Exception("neurons param should be a list")
    if len(neurons) == 0:
        raise Exception("neurons param should have at least one element")
    # Model Building
    if should_flatten:
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(neurons[0], activation="relu")(x)
    else:
        x = tf.keras.layers.Dense(neurons[0], activation="relu")(inputs)
    for i in range(1, len(neurons)):
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(neurons[i], activation="relu")(x)
    output = tf.keras.layers.Dense(output_neuron, activation=output_activation)(x)
    return output


def build_model(inputs, output):
    """
    Build a tensorflow model from input layer and output
    :param inputs: The input layer of the model
    :param output: The output layer of the model (using the functional API)
    :return: A Tensorflow model
    """
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def create_model(input_size, action_size):
    dropout_rate = 0.2
    inputs = build_input_layer(input_size)
    return build_model(
        inputs,
        build_dense_block(inputs, [30, 15, 10], action_size, "softmax", dropout_rate)
    )
