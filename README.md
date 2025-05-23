# Helmet and Object Detection

This project implements a helmet and object detection system using the YOLOv8 model. It leverages Python and PyTorch to detect helmets in images and videos, aiming to enhance safety compliance monitoring in various environments such as construction sites and traffic surveillance.

## Features

* **Helmet Detection**: Identifies whether individuals are wearing helmets in images and videos.
* **Real-time Processing**: Capable of processing video streams for real-time detection.
* **Customizable**: Easily adaptable to detect other objects by retraining the model with different datasets.

## Installation

### Prerequisites

* Python 3.7 or higher
* pip (Python package installer)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Abhay-0103/Helmet-And-Object-Detection.git
   cd Helmet-And-Object-Detection
   ```
2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

*Note: If `requirements.txt` is not provided, you may need to install the necessary packages manually, such as `torch`, `opencv-python`, and `ultralytics`.*

## Usage

### 1. **Helmet Detection on Images**

```bash
python Helmet-Detection.py --image path_to_image.jpg
```

This script will process the specified image and display it with bounding boxes around detected helmets.

### 2. **Helmet Detection on Videos**

```bash
python Helmet-Object-Detection.py --video path_to_video.mp4
```

This script will process the specified video file, performing helmet detection frame by frame and displaying the results in real-time.

### 3. **Real-time Detection via Webcam**

```bash
python Helmet-Object-Detection.py --webcam
```

This will activate your system's webcam and perform real-time helmet detection on the video stream.

### 4. **Using a Custom Trained Model**

If you have a custom-trained YOLOv8 model (e.g., `custom_model.pt`), you can use it by specifying the path:

```bash
python Helmet-Detection.py --image path_to_image.jpg --model path_to_custom_model.pt
```

## Model Files

* **`yolov8n.pt`**: Pre-trained YOLOv8 nano model provided by Ultralytics.
* **`helmet.pt`**: Custom-trained model for helmet detection.([GitHub][2])

*Note: Ensure that the model files are placed in the project directory or specify the correct path when running the scripts.*

## Dataset

The project utilizes a dataset comprising images labeled for helmet detection. If you wish to train your own model:([GitHub][3])

1. **Collect and Label Data**: Use tools like [LabelImg](https://github.com/tzutalin/labelImg) to annotate images.
2. **Prepare Dataset**: Organize the dataset into training and validation sets.
3. **Train the Model**: Use the YOLOv8 training pipeline to train your model with the prepared dataset.([GitHub][2])

*Note: Detailed training instructions are beyond the scope of this README but can be found in the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/).*

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## Acknowledgments

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [PyTorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/)
