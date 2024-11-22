#全局常量定义
neurons_num=32
num_layers=2
frame_parameters=3
epochs = 10 #number of epochs
optimizer_step_value = 0.001 #optimizer step value for pytorch
#测试数据集占总数据集的比例
test_percent = 0.1 
#准确率阈值，当超过阈值就停止训练
break_limit = 94
class_number=12
#手势定义
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
STOP_GESTURE = "stop_gesture"

#给手势进行编码，因为有12个，所有从0~11进行编码
LABELS = {ARM_TO_LEFT: 0, ARM_TO_RIGHT: 1,
            HAND_AWAY:2,  HAND_CLOSE:3,
            CLOSE_FIST_HORIZONTALLY: 4, CLOSE_FIST_PERPENDICULARLY: 5,
            HAND_RIGHT:6, HAND_LEFT:7,
            PALM_DOWN:8, PALM_UP:9,
            HAND_UP: 10, HAND_DOWN: 11}