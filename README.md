# CAP6415_F25_project-Energy_Cost_Prediction_for_Ground_Robots_Using_Multimodal_Sensing

Unmanned ground vehicles (UGV) can be utilized for various purposes including agricultural use, transportation, or environmental monitoring to name a few. Since UGVs, such as the one used in this project, operate on battery power, it can only travel a certain distance before needing a battery recharge. If the battery/energy requirement a UGV needs is unknown, unexpected mission failures or incomplete tasks may occur. Although one can measure the complete distance or time a UGV can travel before running out of battery, it would be time consuming to measure its battery capabilities when running over various surfaces, as the energy consumption will vary depending on each type of surface. In this work, we present a method capable of predicting a UGVs battery/energy consumption when running over various terrains. This will be done by training a neural network on environment data from a camera as well as other sensor data obtained from the UGV when traveling through a specific terrain. To obtain environmental data, a camera will be used to capture images of the terrain that the UGV will be traveling through. Additionally, data from various other sensors located on the UGV, such as inertial measurement unit, GPS data, and battery data will be used as inputs to train the neural network. Our method consists of using a pretrained convolutional neural network, such as ResNet18, for the image data obtained from the camera. Additionally, we will utilize a multilayer perception model to be trained on both the image data and numerical data obtained from the other sensors, in order to predict the UGVs energy consumption when navigating through varying terrains. In this way, a UGVs battery consumption can be predicted and known despite the terrain it travels through. For the purposes of this course project, the camera system and camera data collection along with training the neural network will be the main focus of this project as this is most applicable to the course content. The codes utilized within this project draw inspiration from the code examples found from the following Github page: https://github.com/IntelRealSense/librealsense.git as well as the notebook named “Notebook_03 - RNN and CNN Introduction” provided within the CAP 6415 Computer Vision course. Additionally, due to GitHub file size limitations, the camera image datasets for the training dataset and test dataset are not saved in this repository. Instead, the training camera data can be downloaded from the following Google Drive link: https://drive.google.com/file/d/1UfFbIxSfOAAKuy8xqc__Njx_E6Xbq4us/view?usp=sharing and the test camera data can be downloaded from the following Google Drive link: https://drive.google.com/file/d/1kHXxy-vN67m-23yMomQYs8aTE1n2JRGP/view?usp=sharing. 


## Prerequisites

This project requires installing the following dependencies:

- torch (for deep learning framework)
- torchvision (pretrained CNN model and image transforms)
- numpy (numerical operations)
- pandas (load CSVs and clean up data)
- scikit-learn (split data into train, val, and test sets)
- pillow (load PNG camera images)
- matplotlib (generate and save plots)
- pyrealsense2 (connect to Intel RealSense camera)
- opencv-python (saves camera images)

Install all dependencies using the commands:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Use appropriate CUDA version for your machine
pip install numpy pandas scikit-learn pillow matplotlib opencv-python pyrealsense2


Additional Requirements:

- Python 3.8-3.12
- NVIDIA GPU with CUDA support (recommended for model training) 
- Intel RealSense D435i camera (optional, needed if you are collecting your own image datasets)

If collecting your own image dataset:

- Image CSV file must have the following columns: timestamp (s), filename, label


### This project has the following folder structure:

'''
CAP6415_F25_project-Energy_Cost_Prediction_for_Ground_Robots_Using_Multimodal_Sensing/
  EnergyConsumptionPredictionFiles/
    codes/
      capture_image_data.py                  # script for capturing image data
      NN-code.py                             # script for training the neural network on a training dataset
      test_NN.py                             # script for testing the neural network on a test dataset
      
    data/
      test_dataset/
        camera_data/ 
          11_18_25-camera_data-grassy/
            index.csv                        # CSV for image data
            color_20251118_162954_000.png    # Image data
            ...
            color_20251118_165623_1589.png
            etc.
        test_gps_20251118_162921.csv         # GPS data
        test_grassy_terrain_data.csv         # Telemetry data
      training_dataset/
        camera_data/ 
          11_12_25-camera_data-grassy/
            index.csv                        # CSV for image data
            color_20251112_163125_000.png    # Image data
            ...
            color_20251112_170814_2209.png
            etc.
        train_gps_20251112_163052.csv        # GPS data
        train_grassy_terrain_data.csv        # Telemetry data
      
    model/
      CNN_MLP_model-1.pt                     # saved trained model
    
    results/
      pred_true_plot.png                     # scatter plot showing model performance
      test_run_predictions.csv               # CSV containing model predictions
'''
