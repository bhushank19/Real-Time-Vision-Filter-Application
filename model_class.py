# This file is responsible for loading the models and running inference on them.
# We are using the base code from the opencv_zoo/models/human_segmentation_pphumanseg/pphumanseg.py and opencv_zoo/models/face_detection_yunet/yunet.py files.


import numpy as np
import cv2 as cv

class PPHumanSeg:
    """
    This class is responsible for loading the model and running inference on it for Human Segmentation.
    """
    def __init__(self, modelPath:str, backendId:int=0, targetId:int=0):
        """
        The init function is responsible for loading the model and setting the backend and target.

        Args:
            modelPath (_type_): _description_
            backendId (int, optional): _description_. Defaults to 0.
            targetId (int, optional): _description_. Defaults to 0.
        """
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId
        # This will load the model from the path provided.
        print(self._modelPath)
        self._model = cv.dnn.readNet(self._modelPath)
        # This will set the backend that is going to be used for inference. ex: DNN_BACKEND_INFERENCE_ENGINE, DNN_BACKEND_OPENCV, DNN_BACKEND_HALIDE, DNN_BACKEND_CUDA
        self._model.setPreferableBackend(self._backendId)
        # This is the specific target device we are going to use for inference. ex: DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16, DNN_TARGET_MYRIAD, DNN_TARGET_FPGA, DNN_TARGET_CUDA
        self._model.setPreferableTarget(self._targetId)

        # A empty file name for processing the input.
        self._inputNames = ''
        # The output name of the file that we got from the model.
        self._outputNames = ['save_infer_model/scale_0.tmp_1']
        # The current input size of the image.
        self._currentInputSize = None
        # The input size of the image, we have changed it to be.
        self._inputSize = [192, 192]
        # The mean and standard deviation of the image.
        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId:int, targetId:int)-> None:
        """
        This function is used to change the backend and target of the model.

        Args:
            backendId (int): The integer value of the backend that is going to be used.
            targetId (int): The integer value of the target that is going to be used.
        """
        # Store the backend and target into class object and update the backend and target of the model.
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):
        """_summary_
        Args:
            image (numpy.ndarray): The image that is going to be processed.
        Returns:
            Mat: 4-dimensional Mat with NCHW dimensions order.
        """
        # Convert the image to RGB format.
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Store the current input size of the image.
        self._currentInputSize = image.shape
        # Change it to 192x192
        image = cv.resize(image, (192, 192))
        # We are doing some operations on the image to make it ready for inference.
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        # Re
        return cv.dnn.blobFromImage(image)

    def _postprocess(self, outputBlob):
        """
        This function is used to process the output of the inference basically we are restoring the size and changing type back to report.
        Args:
            outputBlob(): The image we are going to Infer.
        Returns:
            numpy.ndarray: The result of the inference.
        """        
        # Take the first element from the outputBlob. 
        outputBlob = outputBlob[0]
        # Change the shape of the outputBlob to the current input size.
        outputBlob = cv.resize(outputBlob.transpose(1,2,0), (self._currentInputSize[1], self._currentInputSize[0]), interpolation=cv.INTER_LINEAR).transpose(2,0,1)[np.newaxis, ...]
        # Change the type of the outputBlob to report.
        result = np.argmax(outputBlob, axis=1).astype(np.uint8)
        # Return the result.
        return result

    
    def infer(self, image):
        """_summary_
        Args:
            image (MAT): The image we are going to Infer.
        Returns:
            numpy.ndarray: The result of the inference.
        """
        # Preprocess
        inputBlob = self._preprocess(image)

        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward()

        # Postprocess
        results = self._postprocess(outputBlob)

        return results
