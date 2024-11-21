import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(inputs, output, loss, optimizer):
    """
    Build a tensorflow model from input layer and output
    :param inputs: The input layer of the model
    :param output: The output layer of the model (using the functional API)
    :return: A Tensorflow model
    """
    model = Sequential()
    model.add(Dense(64, input_dim=inputs, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output, activation='linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model



# def create_model(input_size, action_size):
#     dropout_rate = 0.2
#     inputs = build_input_layer(input_size)
#     return build_model(
#         inputs,
#         build_dense_block(inputs, [80, 50, 50, 10 ], action_size, "softmax", dropout_rate)
#     )

