"""
author: 木人舟
brief: Define the parameters related to training and testing of the LSTM model. You can modify the training performance by adjusting the parameters in this file.
"""
neurons_num = 32  # Number of neurons in the LSTM network
num_layers = 2  # Number of hidden layers in the LSTM network
frame_parameters = 3  # Number of frame parameters, which are x, y, velocity
epochs = 10  # Number of training epochs
optimizer_step_value = 0.001  # Step value for the PyTorch optimizer
test_percent = 0.2  # Proportion of the test dataset to the total dataset
break_limit = 94  # Accuracy threshold, training stops when the threshold is exceeded
test_percent = 0.2  # Proportion of the test dataset to the total dataset
break_limit = 95  # Accuracy threshold, training stops when the threshold is exceeded
class_number = 12  # Number of classes, there are 12 types of gestures
"""
Definition of 12 types of gestures
"""

ARM_TO_LEFT = "arm_to_left"
ARM_TO_RIGHT = "arm_to_right"
CLOSE_FIST_HORIZONTALLY = "close_fist_horizontally"
CLOSE_FIST_PERPENDICULARLY = "close_fist_perpendicularly"
HAND_CLOSE = "hand_closer"
HAND_AWAY = "hand_away"
HAND_LEFT = "hand_to_left"
HAND_RIGHT = "hand_to_right"
HAND_DOWN = "hand_down"
HAND_UP = "hand_up"
PALM_DOWN = "hand_rotation_palm_down"
PALM_UP = "hand_rotation_palm_up"

"""
Encoding gestures, as there are 12, so encoding them from 0 to 11.
"""
LABELS = {ARM_TO_LEFT: 0, ARM_TO_RIGHT: 1,
          HAND_AWAY: 2, HAND_CLOSE: 3,
            CLOSE_FIST_HORIZONTALLY: 4, CLOSE_FIST_PERPENDICULARLY: 5,
          HAND_RIGHT: 6, HAND_LEFT: 7,
          PALM_DOWN: 8, PALM_UP: 9,
            HAND_UP: 10, HAND_DOWN: 11}
