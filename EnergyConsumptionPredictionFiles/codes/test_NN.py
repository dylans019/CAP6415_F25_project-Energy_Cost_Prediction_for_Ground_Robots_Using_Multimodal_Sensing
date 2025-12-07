# This Python script loads the trained model and tests the model on a test dataset. 

# Dependencies required:
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install pillow
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Note: Use the appropriate CUDA version for your machine. A CPU only version is also available. 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models


# Paths to the testing data
TEST_GPS_CSV_PATH       = "data/test_dataset/test_gps_20251118_162921.csv" 
TEST_TELEMETRY_CSV_PATH = "data/test_dataset/test_grassy_terrain_data.csv" 
TEST_IMAGES_CSV_PATH    = "data/test_dataset/camera_data/11_18_25-camera_data-grassy/index.csv" 
TEST_IMAGE_ROOT_DIR     = "data/test_dataset" 

# Name of the timestamp column for each CSV file
GPS_TIMESTAMP_COL   = "timestamp (sec)"
CAM_TIMESTAMP_COL   = "timestamp (s)"
TELE_TIMESTAMP_COL  = "Timestamp (s)" 

# Important column names for power, speed, and image filename
POWER_COL = "Power (w)"
SPEED_COL = "speed (m/s)"
IMAGE_FILENAME_COL = "filename"

# State features for inputs to the neural network 
STATE_COLS = [
    SPEED_COL,
    "Distance Traveled (m)",
    "up (m)",
    "accel_mps2",
    "cum_dist(m)",
    "heading_sin",
    "heading_cos",
]

# Path to trained model weights 
MODEL_WEIGHTS_PATH = "model/CNN_MLP_model-1.pt"   

BATCH_SIZE = 32 # Number of samples per batch training
MIN_SPEED = 0.05 # Minimum speed threshold, data points below this speed are excluded
RANDOM_SEED = 42 # Seed for random number generation

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



# Function for loading and merging all of the camera and numerical data together 
def load_and_merge_new_csvs():
    gps  = pd.read_csv(TEST_GPS_CSV_PATH)
    tele = pd.read_csv(TEST_TELEMETRY_CSV_PATH)
    cams = pd.read_csv(TEST_IMAGES_CSV_PATH)

    if GPS_TIMESTAMP_COL not in gps.columns:
        raise ValueError(f"GPS CSV missing '{GPS_TIMESTAMP_COL}'")
    if CAM_TIMESTAMP_COL not in cams.columns:
        raise ValueError(f"Camera CSV missing '{CAM_TIMESTAMP_COL}'")
    if TELE_TIMESTAMP_COL not in tele.columns:
        raise ValueError(f"Telemetry CSV missing '{TELE_TIMESTAMP_COL}'")

    # In a unified column named 't', convert the timestamps to seconds (float)
    gps["t"]  = gps[GPS_TIMESTAMP_COL].astype(float)
    tele["t"] = tele[TELE_TIMESTAMP_COL].astype(float)
    cams["t"] = cams[CAM_TIMESTAMP_COL].astype(float)

    gps  = gps.sort_values("t")
    tele = tele.sort_values("t")
    cams = cams.sort_values("t")

    # Merge each camera data to the closest telemetry data to the nearest 0.3 seconds 
    merged = pd.merge_asof(
        cams,
        tele,
        on="t",
        direction="nearest",
        tolerance=0.3 
    )

    # Merge the previously merged camera and telemetry data to the closest gps data to the nearest 0.3 seconds
    merged = pd.merge_asof(
        merged,
        gps,
        on="t",
        direction="nearest",
        tolerance=0.3
    )

    # Drop the rows where power or speed are missing
    before = len(merged)
    merged = merged.dropna(subset=[POWER_COL, SPEED_COL])
    after = len(merged)
    print(f"Dropped {before - after} rows with missing Power/Speed.")

    return merged

# Function to compute the energy per meter (Power/speed) for ground truth labels
def compute_energy_labels(df):

    df = df.copy()

    df[SPEED_COL] = df[SPEED_COL].astype(float)
    df[POWER_COL] = df[POWER_COL].astype(float)

    # If the speed is nearly 0, remove these rows
    df = df[df[SPEED_COL] >= MIN_SPEED].copy()
    # Calculates the energy per meter (Power (W)/speed(m/s))
    df["energy_per_m"] = df[POWER_COL] / df[SPEED_COL]
    # Average 5 time steps centered around each row to reduce noise
    df["energy_per_m"] = (
        df["energy_per_m"]
        .rolling(window=5, min_periods=1, center=True)
        .mean()
    )
    # Drop any rows where label is NaN
    df = df.dropna(subset=["energy_per_m"])

    print(f"Final dataset size after filtering: {len(df)} rows.")
    return df


class JackalEnergyDataset(Dataset):
    def __init__(self, df, image_root, state_cols, label_col="energy_per_m", transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.state_cols = state_cols
        self.label_col = label_col

        # Image transforms
        self.transform = transform or T.Compose([
            T.Resize((224, 224)), # Image resize to 224x224 pixels
            T.ToTensor(), # convert PIL image to PyTorch tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], # Normalize using mean
                        std=[0.229, 0.224, 0.225]), # Normalize using std
        ])

    # Return number of samples in the dataset
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load the image
        img_filename = row[IMAGE_FILENAME_COL]
        img_path = os.path.join(self.image_root, img_filename)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Build the state vector and extract state columns
        state_vals = row[self.state_cols].astype(float).values
        state = torch.tensor(state_vals, dtype=torch.float32)

        # Ground-truth label 
        label = torch.tensor(row[self.label_col], dtype=torch.float32).unsqueeze(0)

        # Return image, state features, and label
        return img, state, label


class EnergyNet(nn.Module):
    """
    Same model used for training:
      - ResNet18 backbone + MLP head
    """

    def __init__(self, num_state_features):
        super().__init__()

        # Load ResNet18 model
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Number of features output by the last ResNet layer
        in_feats = base.fc.in_features
        # Remove classification layer, replace with Identity layer to keep features
        base.fc = nn.Identity()
        self.backbone = base

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(in_feats + num_state_features, 128), # Image features + state features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, state):
        # Obtain image features by passing images through ResNet backbone, results in [batch_size,in_feats]
        img_feat = self.backbone(img)
        # Concatenate image feature vector and state feature vector, results in [batch_size,in_feats+state_dim]
        x = torch.cat([img_feat, state], dim=1)
        # Combined feaures are passed through MLP head, results in [batch_size,1]
        out = self.mlp(x)
        return out


def evaluate_on_new_data():
    # Load and merge the test CSV files
    merged = load_and_merge_new_csvs()

    # Compute the ground-truth labels from Power and speed
    merged = compute_energy_labels(merged)

    # Check if state columns exist
    for col in STATE_COLS:
        if col not in merged.columns:
            raise ValueError(f"State column '{col}' not found in test merged data.")

    # Build dataset and dataloader
    dataset = JackalEnergyDataset(merged, TEST_IMAGE_ROOT_DIR, STATE_COLS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EnergyNet(num_state_features=len(STATE_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()

    # Run predictions
    all_true = []
    all_pred = []
    all_filenames = []

    with torch.no_grad():
        for imgs, states, labels in loader:
            imgs   = imgs.to(device)
            states = states.to(device)
            labels = labels.to(device)

            preds = model(imgs, states) 

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    # Concatenate to 1D arrays
    all_true = np.concatenate(all_true, axis=0).flatten()
    all_pred = np.concatenate(all_pred, axis=0).flatten()

    # Attach the predictions back to the dataframe 
    merged = merged.reset_index(drop=True)
    merged["energy_true"] = all_true
    merged["energy_pred"] = all_pred


    # Compute squared error and absolute error
    errors = all_pred - all_true
    se = errors ** 2
    ae = np.abs(errors)

    # Mean squared error and Root mean squared error 
    mse = np.mean(se)
    rmse = np.sqrt(mse)

    # Mean absolute error
    mae = np.mean(ae)

    # Mean absolute relative error (|pred - true| / |true|)
    denom = np.clip(np.abs(all_true), 1e-6, None)
    are = ae / denom
    mare = np.mean(are)

    # R^2: Coefficient of determination
    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum(se)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    print(
        f"Evaluation Metrics:\n"
        f"  MSE  = {mse:.4f}\n"
        f"  RMSE = {rmse:.4f}\n"
        f"  MAE  = {mae:.4f}\n"
        f"  MARE = {mare:.4f}\n"
        f"  R^2  = {r2:.4f}"
    )

    # Visualize the predicted vs true energy_per_m 
    plt.figure(figsize=(6, 6))
    plt.scatter(all_true, all_pred, s=5, alpha=0.5)
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True energy_per_m (J/m)")
    plt.ylabel("Predicted energy_per_m (J/m)")
    plt.title("Predicted vs True Energy per Meter")
    plt.grid(True)

    #Save results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/pred_true_plot.png", dpi=200)
    plt.close()
    print("Scatter plot is saved to results/pred_true_plot.png")

    out_path = "results/test_run_predictions.csv"
    merged.to_csv(out_path, index=False)
    print(f"Predictions are saved to {out_path}")


if __name__ == "__main__":
    evaluate_on_new_data()
