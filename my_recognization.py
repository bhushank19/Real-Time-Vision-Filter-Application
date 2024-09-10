import numpy as np
import cv2 as cv
from functools import partial
from model_class import PPHumanSeg
import os
import pandas as pd
import mediapipe as mp

class filter_creator(object):
    """
    This class is the used to create all the filters that are going to be used in the project.
    Args:
        object (object): base class from which we are inheriting.    
    """

    def __init__(self):
        self.pp_human_model = PPHumanSeg(modelPath=r'models/human_segmentation_pphumanseg_2023mar.onnx',
                                    backendId=cv.dnn.DNN_BACKEND_OPENCV, targetId=cv.dnn.DNN_TARGET_CPU)
        # self.yunet_model = YuNet('model/face_detection_yunet_2023mar.onnx',backendId=cv.dnn.DNN_BACKEND_OPENCV, targetId=cv.dnn.DNN_TARGET_CPU)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=3,min_detection_confidence=0.5)
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

        # Check if we have to use it or not
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,0,0))
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
        self.mpDraw=mp.solutions.drawing_utils
        # Configure the model
        self._model = self.pp_human_model
        self.weight = 0.6
        self.window = ''
        self.filter_selected = 1
        self.enable_fps = True
        self.background_image_index = 0
        self.deg = 0
        
        self.background_image_path_list = []
        self.kernel_size = 11
        self.func_name = self.create_filter
        self.background_enable = False
        self.mask_csv_list = [] 
        self.img_1 = cv.imread('doctor_image/magic_circle_ccw.png', -1)
        self.img_2 = cv.imread('doctor_image/magic_circle_cw.png', -1)
        
        # Load the first mask image
        self.selected_mask = 0
        self.mask = []
        self.all_landmarks_dict = []
        # self.disable_mask = False
        # key dict
        self.key_mapper = {
            ord('q'): self.q_press,
            ord('f'): self.f_press,
            ord('n'): self.n_press,
            ord('p'): self.p_press,
            ord('1'): self.one_press,
            ord('2'): self.two_press,
            ord('3'): self.three_press,
            ord('4'): self.four_press,
            ord('5'): self.five_press,
            ord('r'): self.rotate_background_image,
            ord('k'): self.change_kernel_size,
            ord('m'): self.change_mask,
            ord('d'): self.d_press,
        }

        #Load all the background images
        self.directory = r'background_images/result'
        for filename in os.listdir(self.directory ):
            f = os.path.join(self.directory , filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.background_image_path_list.append(f)
       
        # Load all the mask images
        for filename in os.listdir(r'mask_images'):
            f = os.path.join(r'mask_images', filename)
            # print(f)
            # checking if it is a file
            if os.path.isfile(f) and f.endswith('.png'):
                mask =  cv.imread(f, cv.IMREAD_UNCHANGED)
                self.mask.append(mask)
                #open the csv_file 
                df = pd.read_csv(f[:-4] + '.csv')
                # Load all the mask csv files
                all_cords = []
                mask_landmarks_dict = {}
                id = []
                for index, row in df.iterrows():
                    landmark_dict = {}
                    cord = []
                    landmark_dict['id'] = int(row.iloc[0])
                    landmark_dict['x'] = int(row.iloc[1])
                    landmark_dict['y'] = int(row.iloc[2])
                    id.append(int(row.iloc[0])) 
                    # append the coordinates [x,y]
                    cord.append([int(row.iloc[1]),int(row.iloc[2])])
                    # Extend the list of all the coordinates
                    all_cords.extend(cord)
                    # Add all the landmarks to the dictionary
                    mask_landmarks_dict[index+1] = landmark_dict
                # Add all the coordinates to the dictionary
                mask_landmarks_dict['cord'] = all_cords
                # Add all the ids to the dictionary
                mask_landmarks_dict['id'] = id
                # Add it to the list of all the landmarks for each mask
                self.mask_csv_list.append(mask_landmarks_dict)          
        
        self.background_image = cv.imread(self.background_image_path_list[self.background_image_index])

    def d_press(self):
        self.background_enable = not self.background_enable
    def q_press(self):
        quit()
        # return "quit"
    def change_mask(self):
        self.selected_mask = self.selected_mask + 1
        if self.selected_mask > len(self.mask)-1:
            self.selected_mask = 0
        
    def f_press(self):
        self.enable_fps = not self.enable_fps
    def n_press(self):
        self.filter_selected = self.filter_selected + 1
        if self.filter_selected > 5:
            self.filter_selected = 1
    def p_press(self):    
        self.filter_selected = self.filter_selected - 1
        if self.filter_selected < 1:
            self.filter_selected = 5
    def one_press(self):
        self.filter_selected = 1
    def two_press(self):
        self.filter_selected = 2
    def three_press(self):
        self.filter_selected = 3
    def four_press(self):
        self.filter_selected = 4
    def five_press(self):
        self.filter_selected = 5
    def rotate_background_image(self):
        self.background_image_index = self.background_image_index + 1
        if self.background_image_index > len(self.background_image_path_list):
            self.background_image_index = 0
    def change_kernel_size(self):
        self.kernel_size = self.kernel_size + 10
        if self.kernel_size > 100:
            self.kernel_size = 11

    def gstreamer_pipeline(self,sensor_id=0,capture_width=1280,capture_height=720,display_width=960,display_height=540,framerate=30,flip_method=0):
        """
        This function is used to create the gstreamer pipeline for the camera.
        Args:
            sensor_id (int, optional): The id of the camera. Defaults to 0.
            capture_width (int, optional): The width of the camera. Defaults to 1920.
            capture_height (int, optional): The height of the camera. Defaults to 1080.
            display_width (int, optional): The width of the display. Defaults to 1920.
            display_height (int, optional): The height of the display. Defaults to 1080.
            framerate (int, optional): The framerate of the camera. Defaults to 30.
            flip_method (int, optional): The flip method of the camera. Defaults to 0.
        Returns:
            str: The gstreamer pipeline.
        """
        return ('nvarguscamerasrc sensor-id =%d!'
                'video/x-raw(memory:NVMM), '
                'width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'nvvidconv flip-method=%d ! '
                'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink' %
                (sensor_id,
                 capture_width,
                 capture_height,
                 framerate,
                 flip_method,
                 display_width,
                 display_height)
                )
    def position_data(lmlist):
        global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip
        wrist = (lmlist[0][0], lmlist[0][1])
        thumb_tip = (lmlist[4][0], lmlist[4][1])
        index_mcp = (lmlist[5][0], lmlist[5][1])
        index_tip = (lmlist[8][0], lmlist[8][1])
        midle_mcp = (lmlist[9][0], lmlist[9][1])
        midle_tip = (lmlist[12][0], lmlist[12][1])
        ring_tip  = (lmlist[16][0], lmlist[16][1])
        pinky_tip = (lmlist[20][0], lmlist[20][1])


    def calculate_distance(p1,p2):
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
        return lenght

    def transparent(frame, targetImg, x, y, size=None):
        if size is not None:
            targetImg = cv.resize(targetImg, size)

        newFrame = frame.copy()
        b, g, r, a = cv.split(targetImg)
        overlay_color = cv.merge((b, g, r))
        mask = cv.medianBlur(a, 1)
        h, w, _ = overlay_color.shape
        roi = newFrame[y:y + h, x:x + w]

        img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(mask))
        img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)
        newFrame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

        return newFrame
 
    def doctor_image(self, frame,background_change = False):
        deg = self.deg
        rgbimg=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result=self.hands.process(rgbimg)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                lmList=[]
                for id, lm in enumerate(hand.landmark):
                    h,w,c=frame.shape
                    coorx, coory=int(lm.x*w), int(lm.y*h)
                    lmList.append([coorx, coory])
                    # cv2.circle(img, (coorx, coory),6,(50,50,255), -1)
                # mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
                filter_creator.position_data(lmList)
                palm = filter_creator.calculate_distance(wrist, index_mcp)
                distance = filter_creator.calculate_distance(index_tip, pinky_tip)
                ratio = distance / palm
                if (ratio > 1.3):
                        centerx = midle_mcp[0]
                        centery = midle_mcp[1]
                        shield_size = 3.0
                        diameter = round(palm * shield_size)
                        x1 = round(centerx - (diameter / 2))
                        y1 = round(centery - (diameter / 2))
                        h, w, c = frame.shape
                        if x1 < 0:
                            x1 = 0
                        elif x1 > w:
                            x1 = w
                        if y1 < 0:
                            y1 = 0
                        elif y1 > h:
                            y1 = h
                        if x1 + diameter > w:
                            diameter = w - x1
                        if y1 + diameter > h:
                            diameter = h - y1
                        shield_size = diameter, diameter
                        ang_vel = 2.0
                        deg = deg + ang_vel
                        if deg > 360:
                            deg = 0
                        hei, wid, col = self.img_1.shape
                        cen = (wid // 2, hei // 2)
                        M1 = cv.getRotationMatrix2D(cen, round(deg), 1.0)
                        M2 = cv.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                        rotated1 = cv.warpAffine(self.img_1, M1, (wid, hei))
                        rotated2 = cv.warpAffine(self.img_2, M2, (wid, hei))
                        if (diameter != 0):
                            frame = filter_creator.transparent(frame,rotated1, x1, y1, shield_size)
                            frame = filter_creator.transparent(frame,rotated2, x1, y1, shield_size)
        self.deg = deg
        return frame

    def create_mask(self, frame, background_change = False):
        """_summary_

        Args:
            frame (_type_): _description_
        """
        # frame = cv.flip(frame, 1)
        RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.face_mesh.process(RGB_frame)
        if background_change:
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)
                return frame
        landmarks_coordinates = []
        

        multiface_cord = result.multi_face_landmarks[0]
        for landmark_of_interest in self.mask_csv_list[self.selected_mask]['id']:
            x = int(multiface_cord.landmark[landmark_of_interest].x*self.w)
            y = int(multiface_cord.landmark[landmark_of_interest].y*self.h)
            landmarks_coordinates.append([x,y])
        
        # Call warp function to apply homography with the face orientation and mask image
        
        return self.warp_image(frame, landmarks_coordinates, self.mask[self.selected_mask], self.mask_csv_list[self.selected_mask]['cord'])
        
    
    def warp_image(self, frame, frame_landmark, mask, mask_landmark):
        """This function is used to warp the image to the face.
        Args:
            frame (image): actual frame
            frame_landmark(list[list]): landmarks of the face detected
            mask (mask_loaded_image): mask image loaded with cv2
            landmarks_coordinates (list[list]): landmarks of the mask
        """
        frame_landmark_np = np.array(frame_landmark,dtype=float)
        mask_landmark_np = np.array(mask_landmark,dtype=float)
        
        # Find the homography matrix
        h, _ = cv.findHomography(mask_landmark_np, frame_landmark_np)
        warped_mask = (cv.warpPerspective(mask, h, (frame.shape[1],frame.shape[0])).astype(float))/255.0
        # Extract the alpha mask of the warped image
        alpha_mask = warped_mask[:,:,3]
        
        # Modify the frame to add the mask
        final_frame = (cv.cvtColor(frame, cv.COLOR_BGR2BGRA).astype(float))/255.0
        for color in range(0, 3):
            final_frame[:,:,color] = alpha_mask*warped_mask[:,:,color]+ (1-alpha_mask)*final_frame[:,:,color]
        
        # final_frame[:,:,:] = cv.erode(final_frame[:,:,:],(5,5),0)
        # final_frame[:,:,:] = cv.GaussianBlur(final_frame[:,:,:],(3,3),0)
        return final_frame

    def create_filter(self, frame, background_change = False):
        """This function will create the background blur and change the background to any image.
        Args:
            video_capture (_type_): _description_
        """
        result = self._model.infer(frame)
        result  = result.reshape((result.shape[1], result.shape[2], result.shape[0]))
        result = np.dstack((result, result, result))
        
        if background_change:
            # Resize background image to match frame size
            background_resized = cv.resize(self.background_image, (frame.shape[1], frame.shape[0]))
            
            final_frame = np.where(result == 0, background_resized, frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return final_frame
        else:
            # Blur the captured frame
            result_blur = cv.GaussianBlur(frame, (self.kernel_size, self.kernel_size),cv.BORDER_DEFAULT)
            final_frame = np.where(result == 0, result_blur, frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return final_frame
            
    def switch_model(self):
        if self.filter_selected == 1 or self.filter_selected == 2:
            self._model = self.pp_human_model
        if self.filter_selected == 3 or self.filter_selected == 4:
            print('Not implemented yet')
           

    def final_method(self):
        """
        This function is used to create the final method that is going to run all the CV windows.
        """
        try:
            self.window_title = 'Filter 1'
            # video_capture = cv.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)
            video_capture = cv.VideoCapture(1)
            if not video_capture.isOpened():
                print("Cannot open camera. Please check the camera connection and try again.")
                exit()
            else:
                # Set the camera resolution
                self.w = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
                self.h = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
                print('Camera resolution: {}x{}'.format(self.w, self.h))
                # Set the tickmeter
                tm = cv.TickMeter()
                previous_filter = 1
                while True:

                    hasFrame, frame = video_capture.read()
                    if not hasFrame:
                        print('No frames grabbed!')
                        break
                    
                    # Handle all the key presses
                    key_pressed = cv.waitKey(1) & 0xFF
                    if key_pressed == 255:
                        pass
                    else: 
                        if key_pressed in self.key_mapper:
                            key_mapper = self.key_mapper[key_pressed]
                            if key_mapper == 'quit':
                                break
                            else:
                                key_mapper()

                    # Switch the model if the filter is changed
                    if previous_filter != self.filter_selected:
                        previous_filter = self.filter_selected
                        self.switch_model()
                        self.window_title = 'Filter ' + str(self.filter_selected)
                        cv.setWindowTitle(winname="Demo", title=self.window_title)
                        if self.filter_selected == 1:
                            self.func_name = self.create_filter
                            self.background_enable = False
                        elif self.filter_selected == 2:
                            self.func_name = self.create_filter
                            self.background_enable = True
                        elif self.filter_selected == 3:
                            result = self.create_filter(frame, background_change = True)
                        elif self.filter_selected == 4:
                            self.func_name = self.create_mask
                            self.background_enable = False
                        elif self.filter_selected == 5:
                            self.func_name = self.doctor_image
                            self.background_enable = False
                        else:
                            pass
                    
                    partial_func = partial(self.func_name, frame)
                    tm.start()
                    result = partial_func(background_change = self.background_enable)
                    tm.stop()
                    
                    if self.enable_fps is not None:
                        cv.putText(result, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv.imshow("Demo", result)
                    tm.reset()

        except KeyboardInterrupt:
                print('Interrupted')

        finally:
            video_capture.release()
            cv.destroyAllWindows()


if __name__ == '__main__':
    filter_creator().final_method()