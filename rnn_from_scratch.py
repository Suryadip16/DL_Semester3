import string

import numpy as np

inputs = np.array([
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
     "X", "Y", "Z"],
    ["Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E", "D",
     "C", "B", "A"],
    ["B", "D", "F", "H", "J", "L", "N", "P", "R", "T", "V", "X", "Z", "A", "C", "E", "G", "I", "K", "M", "O", "Q", "S",
     "U", "W", "Y"],
    ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I",
     "J", "K", "L"],
    ["H", "G", "F", "E", "D", "C", "B", "A", "L", "K", "J", "I", "P", "O", "N", "M", "U", "T", "S", "R", "Q", "X", "W",
     "V", "Z", "Y"]
])

print(inputs.shape)
print(inputs)
expected = np.array([
    ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
     "Y", "Z", "A"],
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
     "X", "Y", "Z"],
    ["C", "E", "G", "I", "K", "M", "O", "Q", "S", "U", "W", "Y", "A", "B", "D", "F", "H", "J", "L", "N", "P", "R", "T",
     "V", "X", "Z"],
    ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
     "K", "L", "M"],
    ["I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E",
     "F", "G", "H"]
])


# A utility function that converts the list of strings into a list of one-hot encoded vectors:


# inputs: np.ndarray -> Asserts that the input to the func should be an ndarray
#  -> np.ndarray ensures that the function returns an ndarray as an output
def string_to_one_hot(inputs: np.ndarray) -> np.ndarray:
    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}
    # string.ascii_uppercase is a constant that contains all uppercase English letters as a string
    # this creates a dict that maps each letter to an index value like "A": 0, "B": 1 etc

    one_hot_inputs = []
    for row in inputs:
        one_hot_list = []
        for character in row:
            if character.upper() in char_to_index:
                one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
                one_hot_vector[char_to_index[character.upper()]] = 1
                #  Find the index for the given character, and append a 1 to that index in one hot vector. Therefore,
                #  every letter in the input will be a one hot encoded vector of size 26 x 1, where its corresponding
                #  index will be a 1 and all other vals will be a 0.
                one_hot_list.append(one_hot_vector)
            one_hot_inputs.append(one_hot_list)
    return np.array(one_hot_inputs)


# Each input sequence has 26 characters and each character (e.g “A”, “B”) will become a list of 26 items, with the item
# matching its index in the alphabet equals to 1 while the rest are 0. So each input sequence will have the shape of
# (26, 26, 1).

class InputLayer:
    inputs: np.ndarray
    U: np.ndarray = None  # U is the weight matrix connecting input to hidden layer.
    delta_U: np.ndarray = None  # delta_U is the gradient calculated during backprop.

    def __init__(self, inputs: np.ndarray, hidden_size: int) -> None:
        self.inputs = inputs
        self.U = np.random.uniform(low=0, high=1, size=(hidden_size, len(inputs[0])))
        self.delta_U = np.zeros_like(self.U)

    def get_input(self, time_step: int) -> np.ndarray:
        return self.inputs[time_step]

    def weighted_sum(self, time_step: int) -> np.ndarray:
        return self.U @ self.get_input(time_step)

    def calculate_deltas_per_step(self, time_step: int, delta_weighted_sum: np.ndarray) -> None:  # (h_dimensions, 1) @ (1, input_size) = (h_dimension, input_size)
        self.delta_U += delta_weighted_sum @ self.get_input(time_step).T

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.U = learning_rate * self.delta_U

    
