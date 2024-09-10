# Video Filter Project

## Overview

This project is a real-time video filtering application using Python. It leverages several libraries including OpenCV, MediaPipe, and NumPy to apply various filters and effects to video streams. The core functionality includes:

- Human segmentation
- Face mesh and face detection
- Real-time filters and effects
- Background image change and mask application

## Prerequisites

Ensure you have the following software installed:

- Python 3.7 or later
- OpenCV
- NumPy
- Pandas
- MediaPipe

You can install the necessary Python packages using pip:

```bash
pip install numpy opencv-python pandas mediapipe
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/video-filter-project.git
   cd video-filter-project
   ```

2. **Download the model files:**

   Ensure that the required model files (`human_segmentation_pphumanseg_2023mar.onnx` and `face_detection_yunet_2023mar.onnx`) and the images (`magic_circle_ccw.png`, `magic_circle_cw.png`) are placed in the appropriate directories as defined in the script:
   
   ```
   Video_filter_JN/models/
   Video_filter_JN/doctor_image/
   ```

3. **Organize Background and Mask Images:**

   Place background images in:
   
   ```
   Video_filter_JN/background_images/result/
   ```

   Place mask images and their corresponding CSV files in:
   
   ```
   Video_filter_JN/mask_images/
   ```

## Usage

1. **Run the script:**

   Execute the script to start the video filter application:

   ```bash
   python filter_creator.py
   ```

2. **Key Controls:**

   - `q`: Quit the application
   - `f`: Toggle FPS display
   - `n`: Switch to the next filter
   - `p`: Switch to the previous filter
   - `1`, `2`, `3`, `4`, `5`: Select specific filters
   - `r`: Rotate the background image
   - `k`: Change kernel size for blurring
   - `m`: Change the mask image
   - `d`: Toggle background image visibility

## Implementation Details

- **Filter Creation**: Utilizes human segmentation and other filters to apply effects.
- **Face Mesh**: Detects facial landmarks to apply masks.
- **Doctor Image Filter**: Applies a shield effect based on hand gestures detected using MediaPipe.

## Troubleshooting

- **Cannot open camera**: Ensure the camera is connected properly and accessible. Check the camera ID or permissions if necessary.
- **Model file errors**: Verify that model files are correctly placed in the specified directories.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

---

Feel free to adjust the instructions and details according to the specific needs of your project and repository.
