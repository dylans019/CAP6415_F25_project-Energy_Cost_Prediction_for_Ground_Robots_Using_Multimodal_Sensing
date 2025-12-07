"""
A source utilized to assist in formulating this code includes the following source:
https://github.com/IntelRealSense/librealsense.git
The specific codes utilized as inspiration includes the code files named "python-tutorial-1-depth.py" and "opencv_pointcloud_viewer.py". 
Both of these code files are found in the github cited above under librealsense/wrappers/python/examples. 
"""

# Required dependencies:
# pip install pyrealsense2
# pip install numpy
# pip install opencv-python

import pyrealsense2 as rs
import numpy as np
import cv2, os, time, csv

# Make folder for saving images
dir_file = "camera_data/camera_data"
os.makedirs(dir_file, exist_ok=True)

# Initialize RealSense camera pipeline
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # color stream set to 640x480 resolution, 30 fps
pipeline.start(cfg)

# CSV path for recording camera data
csv_path = f"{dir_file}/index.csv"

try:
    # Allow auto-exposure to warmup/adjust
    for _ in range(15): # waits for 15 frames 
        pipeline.wait_for_frames()

    # Open CSV file and write header: (timestamp (s), filename, label)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp (s)", "filename", "label"]) # time in seconds starting from 0, image path, label

        # Initialize variables for timing and for the loop
        start = time.time()
        interval = 1.0   # captures an image every 1 second
        shot = 0  # number of images taken
        next_shot_time = start # time of when the next image should be captured

        print("Recording started.")

        while True:
            frames = pipeline.wait_for_frames() # waits for available frames
            c = frames.get_color_frame() # extracts color frame
            if not c:
                continue

            now = time.time() 
            if now >= next_shot_time: # ensure the current time is valid for capturing the next image 
                color = np.asanyarray(c.get_data()) # convert the frame to a numpy array
                ts = time.strftime("%Y%m%d_%H%M%S") # timestamp as year month day_hour minutes seconds
                filename = f"{dir_file}/color_{ts}_{shot:03d}.png"
                cv2.imwrite(filename, color) # save the image to computer as a .png

                t_rel = now - start  # number of seconds from the start
                writer.writerow([f"{t_rel:.2f}", filename, 0]) # write image information to the .csv file
                f.flush()

                print(f"Saved {filename}")
                shot += 1 
                next_shot_time += interval

            time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopped recording.")

finally:
    pipeline.stop() # stop streaming
