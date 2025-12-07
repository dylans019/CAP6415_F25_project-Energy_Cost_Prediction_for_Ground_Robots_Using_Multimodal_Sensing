"""
A source utilized as inspiration to formulate this code includes the notebook titled "Notebook_03 - RNN and CNN Introduction" 
which is provided within the CAP 6415 Computer Vision course. 
"""

# Required dependencies:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Note: Use the appropriate CUDA version for your machine. A CPU only version is also available. 
# pip install pandas
# pip install numpy
# pip install scikit-learn
# pip install pillow

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models


GPS_CSV_PATH = "" # Write path to the GPS CSV file
TELEMETRY_CSV_PATH = "" # Write path to the telemetry CSV file
IMAGES_CSV_PATH = "" # Write path to the camera CSV file
IMAGE_ROOT_DIR = "" # Write path to the camera images saved as .PNG

# Name of the timestamp column for each CSV file
GPS_TIMESTAMP_COL   = "timestamp (sec)" 
CAM_TIMESTAMP_COL   = "timestamp (s)" 
TELE_TIMESTAMP_COL  = "Timestamp (s)" 

# Important column names for power, speed, and image filename
POWER_COL = "Power (w)" 
SPEED_COL = "speed (m/s)" 
IMAGE_FILENAME_COL = "filename" 

# State features for extra neural network inputs
STATE_COLS = [
    SPEED_COL,                    # robot speed
    "Distance Traveled (m)",      # distance traveled from telemetry
    "up (m)",                     # vertical position from GPS
    "accel_mps2",                 # acceleration from GPS
    "cum_dist(m)",                # cumulative travel distance 
    "heading_sin",                # heading direction components
    "heading_cos",
]

BATCH_SIZE = 16   # Number of samples per batch training
NUM_EPOCHS = 10   # Amount of times the model iterates over training dataset
LEARNING_RATE = 1e-4   # How much the model's weights are adjusted while training
RANDOM_SEED = 42   # Seed for random number generation
VAL_SPLIT = 0.15   # Amount of dataset used for validation set
TEST_SPLIT = 0.15   # Amount of dataset used for testing set
MIN_SPEED = 0.05  # Minimum speed threshold, data points below this speed are excluded


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_and_merge_csvs():
    gps  = pd.read_csv(GPS_CSV_PATH)
    tele = pd.read_csv(TELEMETRY_CSV_PATH)
    cams = pd.read_csv(IMAGES_CSV_PATH)

    if GPS_TIMESTAMP_COL not in gps.columns:
        raise ValueError(f"GPS CSV missing '{GPS_TIMESTAMP_COL}'")

    if CAM_TIMESTAMP_COL not in cams.columns:
        raise ValueError(f"Camera CSV missing '{CAM_TIMESTAMP_COL}'")

    if TELE_TIMESTAMP_COL not in tele.columns:
        raise ValueError(f"Telemetry CSV missing '{TELE_TIMESTAMP_COL}'")

    # Create column "t" from the timestamp columns
    gps["t"]  = gps[GPS_TIMESTAMP_COL].astype(float)
    tele["t"] = tele[TELE_TIMESTAMP_COL].astype(float)
    cams["t"] = cams[CAM_TIMESTAMP_COL].astype(float)

    # Sort dataframes by "t" 
    gps  = gps.sort_values("t")
    tele = tele.sort_values("t")
    cams = cams.sort_values("t")

    # In the "t" column, merge each camera image by finding the closest telemetry data row to the nearest 0.3 seconds
    merged = pd.merge_asof(
        cams,
        tele,
        on="t",
        direction="nearest",
        tolerance=0.3 
    )

    # Merge the GPS data to the merged camera and telemetry data to the nearest 0.3 seconds
    merged = pd.merge_asof(
        merged,
        gps,
        on="t",
        direction="nearest",
        tolerance=0.3 
    )

    # If Power or Speed is missing, these rows are dropped
    before = len(merged)
    merged = merged.dropna(subset=[POWER_COL, SPEED_COL])
    after = len(merged)
    print(f"Dropped {before - after} rows with missing Power/Speed.")

    return merged

def compute_energy_labels(df):
    df[SPEED_COL] = df[SPEED_COL].astype(float)
    df[POWER_COL] = df[POWER_COL].astype(float)

    # If the speed is nearly 0, remmove these rows 
    df = df[df[SPEED_COL] >= MIN_SPEED]

    # Calculates energy per meter 
    # J/m = (J/s) / (m/s) = W / (m/s)
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
            T.Normalize(mean=[0.485, 0.456, 0.406], #Normalize using mean
                        std=[0.229, 0.224, 0.225]), #Normalize using std
        ])

    # Return number of samples in the dataset
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Loads image from specified row
        img_filename = row[IMAGE_FILENAME_COL]
        img_path = row[IMAGE_FILENAME_COL]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Build state vector
        # Extract state columns 
        state_vals = row[self.state_cols].astype(float).values
        state = torch.tensor(state_vals, dtype=torch.float32)

        # Label: energy_per_m
        label = torch.tensor(row[self.label_col], dtype=torch.float32).unsqueeze(0)

        return img, state, label #Returns the image, state features, and label

class EnergyNet(nn.Module):

    def __init__(self, num_state_features):
        super().__init__()

        # Load the ResNet18 model 
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

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for imgs, states, labels in loader:
        imgs   = imgs.to(device)
        states = states.to(device)
        labels = labels.to(device)

        # Forward pass:
        preds = model(imgs, states) # model predictions
        loss = nn.functional.mse_loss(preds, labels) #MSE loss between predictions and true labels

        optimizer.zero_grad()

        # Backpropagate: 
        loss.backward()
        optimizer.step()
        # Total loss 
        total_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)

    return total_loss / count # Average loss

def evaluate(model, loader, device):
    model.eval()

    mse_total = 0.0      # sum of squared errors
    mae_total = 0.0      # sum of absolute errors
    mare_total = 0.0     # sum of absolute relative errors
    sum_y = 0.0          # sum of true labels (for R^2)
    sum_y2 = 0.0         # sum of squares of true labels (for R^2)
    count = 0            # number of samples

    with torch.no_grad():
        for imgs, states, labels in loader:
            imgs   = imgs.to(device)
            states = states.to(device)
            labels = labels.to(device)

            # predictions
            preds = model(imgs, states) 

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)

            # Squared error, absolute error
            se = (preds_flat - labels_flat) ** 2
            ae = (preds_flat - labels_flat).abs()

            # Absolute relative error = |pred - true| / |true|
            denom = labels_flat.abs().clamp(min=1e-6)
            are = ae / denom

            mse_total += se.sum().item()
            mae_total += ae.sum().item()
            mare_total += are.sum().item()

            sum_y  += labels_flat.sum().item()
            sum_y2 += (labels_flat ** 2).sum().item()
            count  += labels_flat.numel()

    # Means 
    mse_mean  = mse_total / count # Mean squared error
    mae_mean  = mae_total / count # Mean absolute error
    mare_mean = mare_total / count # Mean absolute relative error
    rmse      = mse_mean ** 0.5 # Root mean squared error

    # R^2: coefficient of determination
    # R^2 = 1 - SS_res / SS_tot
    ss_res = mse_total
    ss_tot = sum_y2 - (sum_y ** 2) / count
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan") 

    return mse_mean, mae_mean, mare_mean, rmse, r2


def main():
    # Merge the CSVs into a single dataframe
    merged = load_and_merge_csvs()

    # Compute the energy_per_m labels 
    merged = compute_energy_labels(merged)

    # Ensures all state columns exist
    for col in STATE_COLS:
        if col not in merged.columns:
            raise ValueError(f"State column '{col}' not found in merged dataframe. "
                             f"Check STATE_COLS list or CSV headers.")

    # Train/val/test split
    # Split the test dataset from the whole dataset 
    df_train_val, df_test = train_test_split(
        merged, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    relative_val = VAL_SPLIT / (1.0 - TEST_SPLIT)
    # Splits the rest of the data into train and validation datasets.
    df_train, df_val = train_test_split(
        df_train_val, test_size=relative_val, random_state=RANDOM_SEED
    )

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    # Datasets and dataloaders
    train_dataset = JackalEnergyDataset(df_train, IMAGE_ROOT_DIR, STATE_COLS)
    val_dataset   = JackalEnergyDataset(df_val,   IMAGE_ROOT_DIR, STATE_COLS)
    test_dataset  = JackalEnergyDataset(df_test,  IMAGE_ROOT_DIR, STATE_COLS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EnergyNet(num_state_features=len(STATE_COLS)).to(device)
    # All model parameters are updated using the learning rate through Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_mse = float("inf")
    os.makedirs("model", exist_ok=True)
    best_model_path = "model/CNN_MLP_model-1.pt"


    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae, val_mare, val_rmse, val_r2 = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE={train_loss:.4f} | "
            f"Val MSE={val_mse:.4f} | "
            f"Val RMSE={val_rmse:.4f} | "
            f"Val MAE={val_mae:.4f} | "
            f"Val MARE={val_mare:.4f} | "
            f"Val R²={val_r2:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")

    # Test evaluation using best model
    print("Loading the model for test evaluation")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_mse, test_mae, test_mare, test_rmse, test_r2 = evaluate(model, test_loader, device) # Evaluate the test data
    print(
        f"Test MSE={test_mse:.4f}, "
        f"Test RMSE={test_rmse:.4f}, "
        f"Test MAE={test_mae:.4f}, "
        f"Test MARE={test_mare:.4f}, "
        f"Test R²={test_r2:.4f}"
    )


if __name__ == "__main__":
    main()
