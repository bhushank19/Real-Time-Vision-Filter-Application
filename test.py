import cv2
print(cv2.__version__)
net = cv2.dnn.readNet('models/human_segmentation_pphumanseg_2023mar.onnx', backendId=cv2.dnn.DNN_BACKEND_OPENCV, targetId=cv2.dnn.DNN_TARGET_CPU)
print("DNN module is working correctly")
