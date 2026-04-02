# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time

# Configuration
FRAG_LEN = 32  # Use 32 historical data points
PRED_LEN = 32  # Predict 32 future points
GAS_FEATURES = 3  # CO, CO2, C2H4

# Add DEFAULT_FRAG_LEN and DEFAULT_PRED_LEN for LSTM compatibility
DEFAULT_FRAG_LEN = FRAG_LEN
DEFAULT_PRED_LEN = PRED_LEN

# Gas network parameters from Gastft.py
GAS_NET_HIDDEN_DIM = 8
GAS_NET_OUTPUT_DIM = 32
GAS_CORRECTION_DIM = 256
MAX_ROLLING_PREDICTION_HORIZON = 500

TRAIN_BATTERIES = ["11_1", "14", "15", "16"]  # Training batteries
TEST_BATTERY = "13"  # Testing battery
BATCH_SIZE = 8
LR = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 500
PATIENCE = 50
CACHE_ROOT = "dataset_cache"
GAS_START = 70  # Start index of gas data to use
GAS_END = 120  # End index of gas data to use
MODEL_PATH = os.path.join("predictors", "hybrid_predictor.pth")  # Default model save/load path
# MODE = "train"  # Run mode: "train" or "predict"
MODE = "predict"  # Run mode: "train" or "predict"
USE_REAL_GAS_DATA = True
# USE_REAL_GAS_DATA = False

ALPHA = 0

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
SEED = 114514
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HybridPredictor(nn.Module):
    """Enhanced LSTM-CNN hybrid predictor with cross-attention and time encoding"""
    def __init__(self, gas_length):
        super(HybridPredictor, self).__init__()
        self.hidden_dim = GAS_NET_OUTPUT_DIM
        
        # 1. Gas feature extraction with enhanced 1D-CNN
        self.gas_cnn = nn.Sequential(
            nn.Conv1d(GAS_FEATURES, GAS_NET_HIDDEN_DIM, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm(gas_length),
            nn.MaxPool1d(2),
            nn.Conv1d(GAS_NET_HIDDEN_DIM, GAS_NET_HIDDEN_DIM, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm(gas_length // 2),
            nn.MaxPool1d(2),
        )
        
        # Channel-wise attention for gas features
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_interaction = nn.Sequential(
            nn.Linear(GAS_NET_HIDDEN_DIM, GAS_NET_HIDDEN_DIM * 2),
            nn.LeakyReLU(),
            nn.Linear(GAS_NET_HIDDEN_DIM * 2, GAS_NET_HIDDEN_DIM),
            nn.Sigmoid()
        )
        
        # Gas feature projection
        self.gas_proj = nn.Sequential(
            nn.Linear(GAS_NET_HIDDEN_DIM * (gas_length // 4), GAS_NET_OUTPUT_DIM * 2),
            nn.LeakyReLU(),
            nn.Linear(GAS_NET_OUTPUT_DIM * 2, GAS_NET_OUTPUT_DIM),
        )
        self.gas_norm = nn.LayerNorm(GAS_NET_OUTPUT_DIM)
        
        # 2. Capacity history processing with BiLSTM
        self.capacity_lstm = nn.LSTM(
            1, GAS_NET_OUTPUT_DIM // 2, bidirectional=True, batch_first=True
        )
        self.capacity_proj = nn.Linear(GAS_NET_OUTPUT_DIM, GAS_NET_OUTPUT_DIM)
        self.capacity_norm = nn.LayerNorm(GAS_NET_OUTPUT_DIM)
        
        # 3. Cross-attention for feature fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=GAS_NET_OUTPUT_DIM, num_heads=4, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(GAS_NET_OUTPUT_DIM)
        self.attn_proj = nn.Linear(GAS_NET_OUTPUT_DIM, GAS_NET_OUTPUT_DIM)

        # 4. Time encoding for future relative time
        self.time_embedding = nn.Embedding(
            num_embeddings=MAX_ROLLING_PREDICTION_HORIZON, embedding_dim=GAS_NET_OUTPUT_DIM
        )
        self.time_norm = nn.LayerNorm(GAS_NET_OUTPUT_DIM)
        
        # 5. Fusion LSTM to process combined features
        self.fusion_lstm = nn.LSTM(
            input_size=GAS_NET_OUTPUT_DIM * 2, hidden_size=GAS_CORRECTION_DIM, batch_first=True
        )
        
        # 6. Dual prediction heads for rho and theta
        self.rho_head = nn.Sequential(
            nn.Linear(GAS_CORRECTION_DIM, GAS_CORRECTION_DIM // 2),
            nn.ReLU(),
            nn.Linear(GAS_CORRECTION_DIM // 2, 1)
        )
        
        self.theta_head = nn.Sequential(
            nn.Linear(GAS_CORRECTION_DIM, GAS_CORRECTION_DIM // 2),
            nn.ReLU(),
            nn.Linear(GAS_CORRECTION_DIM // 2, 1)
        )
    
    def forward(self, gas_data, capacity_data, target_length=None, use_real_gas=True):
        # gas_data: [batch, gas_features, gas_length]
        # capacity_data: [batch, frag_len]
        # target_length: optional, if provided, generate predictions up to this length
        # use_real_gas: whether to use real gas data or all-ones
        batch_size = gas_data.size(0)
        
        # Determine prediction length
        pred_len = target_length if target_length is not None else PRED_LEN
        
        # Process gas data with enhanced CNN
        gas_features = self.gas_cnn(gas_data)  # [batch, hidden_dim, gas_length//4]
        
        # Replace gas data with all-ones if use_real_gas is False
        if not use_real_gas:
            gas_features = torch.zeros_like(gas_features)

        # Apply channel-wise attention
        global_features = self.global_avg_pool(gas_features).squeeze(-1)  # [batch, hidden_dim]
        interaction_weights = self.feature_interaction(global_features)  # [batch, hidden_dim]
        gas_features = gas_features * interaction_weights.unsqueeze(2)  # [batch, hidden_dim, gas_length//4]
        
        # Flatten and project gas features
        gas_features = gas_features.view(batch_size, -1)  # [batch, hidden_dim * gas_length//4]
        gas_features = self.gas_proj(gas_features)  # [batch, output_dim]
        gas_features = gas_features.unsqueeze(1)  # [batch, 1, output_dim]
        gas_features = self.gas_norm(gas_features)
        
        # Process capacity data with BiLSTM
        capacity_data = capacity_data.unsqueeze(-1)  # [batch, frag_len, 1]
        lstm_out, _ = self.capacity_lstm(capacity_data)  # [batch, frag_len, hidden_dim*2]
        
        # Take the last output and project
        capacity_features = self.capacity_proj(lstm_out)  # [batch, hidden_dim]
        capacity_features = self.capacity_norm(capacity_features)

        # Apply cross-attention fusion
        attn_out, _ = self.cross_attn(gas_features, capacity_features, capacity_features)
        fused_features = self.attn_norm(gas_features + attn_out)  # [batch, 1, hidden_dim]
        fused_features = self.attn_proj(fused_features)  # [batch, 1, output_dim]
        fused_features = fused_features.squeeze(1)  # [batch, output_dim]
        
        # Create time indices for prediction horizon - start from 1 instead of 0
        time_indices = torch.arange(1, pred_len + 1, device=gas_data.device)  # [pred_len]
        time_indices = time_indices.repeat(batch_size, 1)  # [batch, pred_len]
        
        # Get time embeddings
        time_emb = self.time_embedding(time_indices)  # [batch, pred_len, hidden_dim]
        time_emb = self.time_norm(time_emb)  # [batch, pred_len, hidden_dim]
        
        # Expand fused features for all time steps
        fused_expanded = fused_features.unsqueeze(1).repeat(1, pred_len, 1)  # [batch, pred_len, hidden_dim]
        
        # Combine features with time embeddings
        combined = torch.cat([fused_expanded, time_emb], dim=-1)  # [batch, pred_len, hidden_dim*2]
        
        # Process with fusion LSTM
        lstm_out, _ = self.fusion_lstm(combined)  # [batch, pred_len, hidden_dim]
        
        # Generate predictions with dual heads
        rho = self.rho_head(lstm_out).squeeze(-1)  # [batch, pred_len]
        theta = self.theta_head(lstm_out).squeeze(-1)  # [batch, pred_len]
        
        # Get initial capacity (C0) from the last point of historical data
        C0 = capacity_data[:, -1, 0]  # [batch]
        C0_expanded = C0.unsqueeze(1).repeat(1, pred_len)  # [batch, pred_len]
        
        # Final prediction: C0 - (rho * C0 + theta)
        pred = C0_expanded - (rho * C0_expanded + theta)  # [batch, pred_len]
        
        # Apply physical constraints
        pred = torch.clamp(pred, min=0.0, max=1.1)
        
        # Return only pred, no need for rho and theta
        return pred

# Data loading utilities
def load_battery_summary(battery_id):
    """Load battery summary data"""
    file_path = os.path.join(CACHE_ROOT, f"cell{battery_id}.npz")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return None
    
    try:
        data = np.load(file_path)
        return {
            "cycle": data["cycle"],
            "QDischarge": data["QDischarge"] / 100.0
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_gas_data(battery_id, cycle_num):
    """Load gas data for a specific cycle"""
    # Format: cell{battery_id}_cycles/cycle_xxx.npz
    cycles_dir = os.path.join(CACHE_ROOT, f"cell{battery_id}_cycles")
    cycle_file = os.path.join(cycles_dir, f"cycle_{cycle_num:03d}.npz")
    
    if not os.path.exists(cycle_file):
        return None
    
    try:
        data = np.load(cycle_file)      
        gas_values = []
        target_length = 200  # Target length after padding
        
        # Load all three gas types
        for gas_type in ['CO', 'CO2', 'C2H4']:
            key = f"{gas_type}"
            if key in data:
                gas_conc = data[key]
                
                # Step 1: Pad data to target length if necessary
                current_len = len(gas_conc)
                if current_len < target_length:
                    pad_total = target_length - current_len
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    gas_conc = np.pad(gas_conc, (pad_left, pad_right), 'constant', constant_values=0)
                elif current_len > target_length:
                    # If longer than target, trim to target length
                    gas_conc = gas_conc[:target_length]
                
                # Step 2: Extract indices 9-189, excluding index 99 (total 180 points)
                indices = list(range(9, 99)) + list(range(100, 190))
                gas_conc = gas_conc[indices]
                
                # Step 3: Z-score normalization on the 180 points
                mean = np.mean(gas_conc)
                std = np.std(gas_conc)
                gas_normalized = (gas_conc - mean) / (std + 1e-8)
                
                # Step 4: Select GAS_START to GAS_END range
                gas_selected = gas_normalized[GAS_START:GAS_END]
                
                gas_values.append(gas_selected)
            else:
                # If gas data is missing, create zeros array with the selected length
                gas_values.append(np.zeros(GAS_END - GAS_START))
        
        return np.stack(gas_values)
    except Exception as e:
        print(f"Error loading gas data for battery {battery_id}, cycle {cycle_num}: {e}")
        return None

def prepare_data(batteries):
    """Prepare training data with battery ID tracking"""
    gas_data = []
    capacity_history = []
    capacity_future = []
    battery_ids = []  # Track which battery each sample comes from
    
    for battery_id in batteries:
        print(f"Processing battery {battery_id}...")
        
        # Load summary data
        summary = load_battery_summary(battery_id)
        if not summary:
            continue
        
        qd = summary["QDischarge"]
        cycles = summary["cycle"]
        
        # Load metadata to get available cycles
        meta_file = os.path.join(CACHE_ROOT, f"cell{battery_id}.json")
        if not os.path.exists(meta_file):
            continue
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        available_cycles = set(info['cycle_number'] for info in meta.get('cycles_info', []))
        
        # Create training samples for each available gas cycle
        for i in range(len(qd) - FRAG_LEN):
            # Get cycle number for gas data
            cycle_idx = i + FRAG_LEN
            if cycle_idx >= len(cycles):
                continue
            
            cycle_num = int(cycles[cycle_idx])
            if cycle_num not in available_cycles:
                continue
            
            # Load gas data
            gas = load_gas_data(battery_id, cycle_num)
            if gas is None:
                continue
            
            # Get historical capacity (input)
            hist = qd[i:i+FRAG_LEN]
            
            # Get future capacity (target): from current cycle to end of battery life
            future = qd[i+FRAG_LEN:]
            
            # Only add samples with sufficient future data
            if len(future) < 1:
                continue
            
            # Add to training data
            gas_data.append(gas)
            capacity_history.append(hist)
            capacity_future.append(future)
            battery_ids.append(battery_id)  # Track battery ID
    
    if not gas_data:
        print("Error: No training data prepared!")
        return None, None, None, None
    
    print(f"Prepared data:")
    print(f"  Number of samples: {len(gas_data)}")
    print(f"  Gas data shape: {np.array(gas_data).shape}")
    print(f"  Capacity history shape: {np.array(capacity_history).shape}")
    print(f"  Future capacity samples: {len(capacity_future)} (varying lengths)")
    print(f"  Battery IDs: {set(battery_ids)}")
    # Print battery distribution
    for battery in set(battery_ids):
        count = battery_ids.count(battery)
        print(f"  Battery {battery}: {count} samples")
    
    return gas_data, capacity_history, capacity_future, battery_ids

def time_position_weighted_loss(pred, target, time_indices, alpha=0.1):
    """Weighted loss function that considers time position with specified weight formula"""
    
    batch_size, pred_len = pred.shape
    
    # Calculate time weights using the specified formula
    max_time = time_indices.max()
    weights = 1.0 + (alpha * torch.abs(time_indices - 0.5 * max_time) / (0.5 * max_time))
    
    # Calculate weighted MSE loss directly between pred and target
    mse_loss = weights * ((pred - target) ** 2)
    mse_loss = torch.mean(mse_loss)
    
    # No regularization, just return the weighted MSE loss
    return mse_loss

def train_model(train_gas, train_hist, train_future, val_gas, val_hist, val_future, fold=0):
    """Train the hybrid predictor with variable length predictions"""
    # Convert to numpy arrays
    train_gas = np.array(train_gas)
    train_hist = np.array(train_hist)
    
    val_gas = np.array(val_gas)
    val_hist = np.array(val_hist)
    
    # Initialize model
    gas_length = train_gas.shape[-1]
    model = HybridPredictor(gas_length=gas_length).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Add Cosine Annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS // 5,  # Maximum number of iterations for each cycle
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 0
    best_model = None
    
    print("\nTraining model...")
    
    # Convert to tensors for validation (we'll process training samples individually)
    val_gas_tensor = torch.tensor(val_gas, dtype=torch.float32).to(device)
    val_hist_tensor = torch.tensor(val_hist, dtype=torch.float32).to(device)
    
    # Start time tracking
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Use random shuffling for training
        indices = np.arange(len(train_gas))
        np.random.shuffle(indices)
        
        for i in indices:
            optimizer.zero_grad()
            
            # Get single sample
            gas_sample = torch.tensor(train_gas[i:i+1], dtype=torch.float32).to(device)
            hist_sample = torch.tensor(train_hist[i:i+1], dtype=torch.float32).to(device)
            future_sample = train_future[i]
            
            # Forward pass with dynamic prediction length
            pred_length = len(future_sample)
            pred = model(gas_sample, hist_sample, target_length=pred_length)
            
            # Convert future sample to tensor
            future_tensor = torch.tensor(future_sample, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Create time indices for loss calculation
            time_indices = torch.arange(1, pred_length + 1, device=gas_sample.device)
            time_indices = time_indices.unsqueeze(0).repeat(pred.shape[0], 1)  # [batch, pred_len]
            
            # Calculate loss using the specified weight formula
            loss = time_position_weighted_loss(pred, future_tensor, time_indices, alpha=ALPHA)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_gas)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(len(val_gas)):
                gas_sample = val_gas_tensor[i:i+1]
                hist_sample = val_hist_tensor[i:i+1]
                future_sample = val_future[i]
                
                # Forward pass with dynamic prediction length
                pred_length = len(future_sample)
                pred = model(gas_sample, hist_sample, target_length=pred_length)
                
                # Convert future sample to tensor
                future_tensor = torch.tensor(future_sample, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Create time indices for loss calculation
                time_indices = torch.arange(1, pred_length + 1, device=gas_sample.device)
                time_indices = time_indices.unsqueeze(0).repeat(pred.shape[0], 1)  # [batch, pred_len]
                
                # Calculate validation loss using the same weighted MSE loss
                val_loss += time_position_weighted_loss(pred, future_tensor, time_indices, alpha=ALPHA).item()
        
        val_loss /= len(val_gas)
        
        # Update learning rate with Cosine Annealing scheduler
        scheduler.step()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_model = model.state_dict().copy()
            print("  Validation loss improved, saving best model state...")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    epochs_run = epoch + 1
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
        # Create predictors folder if it doesn't exist
        os.makedirs("predictors", exist_ok=True)
        
        # Save the best model to file with appropriate name
        if fold > 0:  # Cross-validation mode
            save_path = os.path.join("predictors", f"hybrid_predictor_fold_{fold}.pth") if USE_REAL_GAS_DATA else os.path.join("predictors", f"compared_predictor_fold_{fold}.pth")
            print(f"\nBest model for fold {fold} saved to {save_path}")
        else:  # Regular training mode
            save_path = os.path.join("predictors", "hybrid_predictor.pth") if USE_REAL_GAS_DATA else os.path.join("predictors", "compared_predictor.pth")
            print(f"\nBest model saved to {save_path}")
        torch.save(model.state_dict(), save_path)
    
    # Print training statistics
    print(f"\nTraining completed in {training_time:.2f} seconds, {epochs_run} epochs")
    
    # Return model along with training statistics
    return model, training_time, epochs_run

def load_model(model_path):
    """Load a pre-trained hybrid model"""
    # Determine gas_length from GAS_START and GAS_END
    gas_length = GAS_END - GAS_START
    model = HybridPredictor(gas_length=gas_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"\nLoaded pre-trained hybrid model from {model_path}")
    return model

class LSTMModel(nn.Module):
    """Enhanced LSTM Model for capacity prediction with deeper architecture"""
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, output_size=DEFAULT_PRED_LEN, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with deeper architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Deeper dense layers with residual connections to better capture long-term trends
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout and deeper dense layers with residual connections
        out_res = out
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

def create_lstm_scaler():
    """Create the exact same scaler used in 02_LSTM_Predictor.py"""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Set fixed min and max values based on battery capacity physics
    fixed_min = 0.5
    fixed_max = 1.1
    # Fit scaler with fixed min and max instead of using actual data range
    scaler.fit([[fixed_min], [fixed_max]])
    return scaler

# Direct copy of the predict_segment function from 02_LSTM_Predictor.py
def predict_segment(model, input_segment, scaler, pred_len=DEFAULT_PRED_LEN):
    """Predict future cycles based on input segment
    
    Args:
        model: trained LSTM model
        input_segment: numpy array of shape (frag_len,) containing input data
        scaler: MinMaxScaler used for normalization
        pred_len: number of cycles to predict
        
    Returns:
        predictions: numpy array of shape (pred_len,) containing predicted values
    """
    model.eval()
    
    # Normalize input segment
    input_segment_scaled = scaler.transform(input_segment.reshape(-1, 1))
    
    # Convert to PyTorch tensor and reshape to [batch_size, seq_len, features] = [1, frag_len, 1]
    input_tensor = torch.tensor(input_segment_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make prediction - output shape is [1, 32] (1 batch, 32 predictions)
    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()
    
    # CRITICAL FIX: Reshape from [1, 32] to [32, 1] before inverse_transform
    # The scaler expects [n_samples, 1] where each sample is a single prediction
    prediction_scaled_reshaped = prediction_scaled.reshape(-1, 1)
    
    # Inverse transform to get actual values - now correctly shaped as [32, 1]
    prediction = scaler.inverse_transform(prediction_scaled_reshaped)
    prediction = prediction.flatten()  # Convert to [32,]
    
    # Apply physical constraints based on historical degradation patterns
    for i in range(len(prediction)):
        # Ensure capacity is within physically possible range
        prediction[i] = max(prediction[i], 0.0)
        prediction[i] = min(prediction[i], 1.1)
    
    return prediction

def lstm_rolling_prediction(lstm_model, hist_data, pred_steps):
    """Rolling prediction using LSTM model with EXACT same logic as 02_LSTM_Predictor.py"""
    # Create the exact same scaler used in 02_LSTM_Predictor.py
    scaler = create_lstm_scaler()
    
    all_predictions = []
    current_input = hist_data.copy()
    target_length = pred_steps
    
    # Keep predicting until we reach or exceed the target length
    while len(all_predictions) < target_length:
        # Predict next chunk (always predict DEFAULT_PRED_LEN cycles, just like in 02_LSTM_Predictor.py)
        pred_chunk = predict_segment(lstm_model, current_input, scaler, pred_len=DEFAULT_PRED_LEN)
        
        # Calculate how many predictions we need to add this iteration
        remaining = target_length - len(all_predictions)
        take_count = min(DEFAULT_PRED_LEN, remaining)
        
        # Add only the needed predictions to the list
        all_predictions.extend(pred_chunk[:take_count])
        
        # If we've reached the target length, break
        if len(all_predictions) >= target_length:
            break
        
        # Update input with FULL DEFAULT_PRED_LEN predictions for next iteration
        # This maintains consistent rolling prediction mechanism as in 02_LSTM_Predictor.py
        current_input = np.concatenate([current_input[DEFAULT_PRED_LEN:], pred_chunk])
    
    # Apply physical constraints to all predictions, just to be safe
    all_predictions = np.array(all_predictions)
    all_predictions = np.clip(all_predictions, 0.0, 1.1)
    
    return all_predictions

def validate_with_comparison(hybrid_model, battery_id, is_ensemble=False):
    """Validate hybrid model and compare with LSTM model, save plots"""
    print(f"\nValidating and comparing models on battery {battery_id}...")
    
    # Load test data
    summary = load_battery_summary(battery_id)
    if not summary:
        return
    
    qd = summary["QDischarge"]
    cycles = summary["cycle"]
    
    # Get available cycles
    meta_file = os.path.join(CACHE_ROOT, f"cell{battery_id}.json")
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    available_cycles = set(info['cycle_number'] for info in meta.get('cycles_info', []))
    
    # Load LSTM model
    lstm_model_path = "lstm_predictor.pth"
    lstm_model = None
    if os.path.exists(lstm_model_path):
        # Create LSTM model instance with the exact same parameters as in 02_LSTM_Predictor.py
        lstm_model = LSTMModel(
            input_size=1, 
            hidden_size=256, 
            num_layers=4, 
            output_size=DEFAULT_PRED_LEN, 
            dropout=0.1
        )
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        lstm_model.eval()
        print(f"\nLoaded pre-trained LSTM model from {lstm_model_path}")
    else:
        print(f"Warning: LSTM model file {lstm_model_path} not found. Skipping LSTM comparison.")
    
    # Load all Compared fold models for ensemble prediction
    compared_models = []
    compared_model_paths = [os.path.join("predictors", f"compared_predictor_fold_{fold}.pth") for fold in range(1, 6)]
    
    # Filter existing model files
    existing_compared_paths = [path for path in compared_model_paths if os.path.exists(path)]
    
    if existing_compared_paths:
        print(f"Loaded {len(existing_compared_paths)} Compared fold models for ensemble prediction")
        # Only print parameters for the first compared model, skip others
        for path in existing_compared_paths:
            compared_model = load_model(path)
            compared_models.append(compared_model)
    else:
        print(f"Warning: No Compared fold model files found. Skipping Compared model comparison.")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("test_results", f"cell{battery_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Excel writer for saving results
    excel_file_path = os.path.join("test_results", f"cell{battery_id}_result.xlsx")
    excel_writer = pd.ExcelWriter(excel_file_path, engine='openpyxl')
    
    # Test model on multiple starting points
    all_metrics = []
    sample_count = 0
    
    for i in range(len(qd) - FRAG_LEN):
        # Get historical data
        hist = qd[i:i+FRAG_LEN]
        
        # Get true future (from current cycle to end)
        true_future = qd[i+FRAG_LEN:]
        
        # Skip if no future data
        if len(true_future) < 1:
            continue
        
        # Get gas data
        cycle_idx = i + FRAG_LEN
        if cycle_idx >= len(cycles):
            continue
        
        cycle_num = int(cycles[cycle_idx])
        if cycle_num not in available_cycles:
            continue
        
        gas = load_gas_data(battery_id, cycle_num)
        if gas is None:
            continue
        
        # Make prediction with hybrid model
        gas_tensor = torch.tensor(gas, dtype=torch.float32).unsqueeze(0).to(device)
        hist_tensor = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(device)
        
        hybrid_pred = None
        hybrid_std = None
        
        if is_ensemble:
            # Ensemble prediction with uncertainty quantification using fold models
            fold_predictions = []
            for model in hybrid_model:
                with torch.no_grad():
                    pred = model(gas_tensor, hist_tensor, target_length=len(true_future))
                fold_predictions.append(pred.cpu().numpy()[0])
            
            # Calculate mean and standard deviation of fold predictions
            fold_predictions = np.array(fold_predictions)
            hybrid_pred = np.mean(fold_predictions, axis=0)
            hybrid_std = np.std(fold_predictions, axis=0)
        else:
            # Single model prediction
            with torch.no_grad():
                pred = hybrid_model(gas_tensor, hist_tensor, target_length=len(true_future))
            hybrid_pred = pred.cpu().numpy()[0]
        
        # Make prediction with LSTM model using rolling prediction
        lstm_pred = None
        if lstm_model is not None:
            lstm_pred = lstm_rolling_prediction(lstm_model, hist, len(true_future))
        
        # Make prediction with Compared model (ensemble of all fold models)
        compared_pred = None
        if compared_models:
            # Create a copy of the gas tensor to avoid modifying the original
            compared_gas_tensor = gas_tensor.clone()
            # Use use_real_gas=False to get all-ones gas data
            fold_predictions = []
            for compared_model in compared_models:
                with torch.no_grad():
                    pred = compared_model(compared_gas_tensor, hist_tensor, target_length=len(true_future), use_real_gas=False)
                fold_predictions.append(pred.cpu().numpy()[0])
            
            # Calculate ensemble prediction as the mean of all fold predictions
            fold_predictions = np.array(fold_predictions)
            compared_pred = np.mean(fold_predictions, axis=0)
        
        # Calculate metrics for hybrid model
        hybrid_mse = mean_squared_error(true_future, hybrid_pred)
        hybrid_mae = mean_absolute_error(true_future, hybrid_pred)
        hybrid_rmse = np.sqrt(hybrid_mse)
        
        metrics_entry = {
            'cycle': cycle_num,
            'hybrid_mse': hybrid_mse,
            'hybrid_mae': hybrid_mae,
            'hybrid_rmse': hybrid_rmse
        }
        
        # Calculate metrics for LSTM model if available
        if lstm_pred is not None:
            lstm_mse = mean_squared_error(true_future, lstm_pred)
            lstm_mae = mean_absolute_error(true_future, lstm_pred)
            lstm_rmse = np.sqrt(lstm_mse)
            
            metrics_entry['lstm_mse'] = lstm_mse
            metrics_entry['lstm_mae'] = lstm_mae
            metrics_entry['lstm_rmse'] = lstm_rmse
        
        # Calculate metrics for Compared model if available
        if compared_pred is not None:
            compared_mse = mean_squared_error(true_future, compared_pred)
            compared_mae = mean_absolute_error(true_future, compared_pred)
            compared_rmse = np.sqrt(compared_mse)
            
            metrics_entry['compared_mse'] = compared_mse
            metrics_entry['compared_mae'] = compared_mae
            metrics_entry['compared_rmse'] = compared_rmse
        
        # Print metrics
        print(f"  Cycle {cycle_num}: Hybrid RMSE={hybrid_rmse:.6f}")
        if lstm_pred is not None:
            print(f"    LSTM RMSE={lstm_rmse:.6f}")
        if compared_pred is not None:
            print(f"    Compared RMSE={compared_rmse:.6f}")
        
        all_metrics.append(metrics_entry)
        
        # Plot and save results with all three models
        plot_prediction_comparison(
            hist, true_future, hybrid_pred, lstm_pred, compared_pred, cycle_num, battery_id, output_dir, hybrid_std
        )
        
        # Save results to Excel file
        # Use cycle_xxx as sheet name, where xxx is the gas sampling cycle number with 3 digits
        sheet_name = f"cycle_{cycle_num:03d}"
        
        # Create data for Excel sheet
        # Get actual cycle numbers from cycles array
        future_cycle_nums = cycles[i+FRAG_LEN:i+FRAG_LEN+len(true_future)]
        future_cycle_nums = [int(c) for c in future_cycle_nums]
        
        # Create DataFrame with uncertainty information
        data = {
            'Cycle': future_cycle_nums,
            'True Future Capacity': true_future,
            'LSTM Predictor': lstm_pred if lstm_pred is not None else [None]*len(true_future),
            'Hybrid Predictor': hybrid_pred,
            'Compared Predictor': compared_pred if compared_pred is not None else [None]*len(true_future)
        }
        
        # Add uncertainty columns if available
        if hybrid_std is not None:
            data['Hybrid Std Dev'] = hybrid_std
            data['Hybrid Lower (95% CI)'] = hybrid_pred - 2 * hybrid_std
            data['Hybrid Upper (95% CI)'] = hybrid_pred + 2 * hybrid_std
        
        df = pd.DataFrame(data)
        
        # Write to Excel sheet
        df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
    
    # Print average metrics
    if all_metrics:
        avg_hybrid_mse = np.mean([m['hybrid_mse'] for m in all_metrics])
        avg_hybrid_mae = np.mean([m['hybrid_mae'] for m in all_metrics])
        avg_hybrid_rmse = np.mean([m['hybrid_rmse'] for m in all_metrics])
        
        print(f"\nAverage Metrics for Hybrid Model:")
        print(f"  MSE: {avg_hybrid_mse:.6f}")
        print(f"  MAE: {avg_hybrid_mae:.6f}")
        print(f"  RMSE: {avg_hybrid_rmse:.6f}")
        
        if 'lstm_rmse' in all_metrics[0]:
            avg_lstm_mse = np.mean([m['lstm_mse'] for m in all_metrics])
            avg_lstm_mae = np.mean([m['lstm_mae'] for m in all_metrics])
            avg_lstm_rmse = np.mean([m['lstm_rmse'] for m in all_metrics])
            
            print(f"\nAverage Metrics for LSTM Model:")
            print(f"  MSE: {avg_lstm_mse:.6f}")
            print(f"  MAE: {avg_lstm_mae:.6f}")
            print(f"  RMSE: {avg_lstm_rmse:.6f}")
    
    # Save and close Excel writer
    excel_writer.close()
    print(f"\nPrediction results saved to Excel file: {excel_file_path}")
    
    print(f"\nValidation completed for battery {battery_id}")
    print(f"Prediction plots saved in: {output_dir}")

def plot_prediction_comparison(hist, true_future, hybrid_pred, lstm_pred, compared_pred, cycle_num, battery_id, output_dir, hybrid_std=None):
    """Plot prediction comparison and save to file with uncertainty quantification"""
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    hist_cycles = np.arange(len(hist))
    future_cycles = np.arange(len(hist), len(hist) + len(true_future))
    
    plt.plot(hist_cycles, hist, 'b-', label='Historical Capacity')
    plt.plot(future_cycles, true_future, 'g-', label='True Future Capacity')
    
    # Plot Hybrid Predictor with purple color and uncertainty range if available
    if hybrid_std is not None:
        plt.plot(future_cycles, hybrid_pred, 'm--', label='Hybrid Predictor')
        # Plot uncertainty range (2 standard deviations)
        plt.fill_between(future_cycles, 
                        hybrid_pred - 2 * hybrid_std, 
                        hybrid_pred + 2 * hybrid_std, 
                        color='purple', alpha=0.2, label='95% Uncertainty Range')
    else:
        plt.plot(future_cycles, hybrid_pred, 'm--', label='Hybrid Predictor')
    
    # Plot LSTM Predictor with red color if available
    if lstm_pred is not None:
        plt.plot(future_cycles, lstm_pred, 'r--', label='LSTM Predictor')
    
    # Plot Compared Predictor with orange/yellow color if available
    if compared_pred is not None:
        plt.plot(future_cycles, compared_pred, 'y--', label='Compared Predictor (All-Ones Gas)')
    
    # Add vertical line at prediction start
    plt.axvline(x=len(hist)-1, color='k', linestyle='--', label='Prediction Start')
    
    # Add labels and title
    plt.xlabel('Cycle')
    plt.ylabel('Normalized Capacity')
    plt.title(f'Battery {battery_id} - Cycle {cycle_num} Prediction Comparison')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_filename = os.path.join(output_dir, f'prediction_cycle_{cycle_num:03d}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved prediction plot: {plot_filename}")

def main():
    """Main function controlled by MODE configuration"""
    print("Starting LSTM-CNN Hybrid Battery Capacity Predictor...")
    print(f"Current mode: {MODE}")
    
    model = None
    
    if MODE == "train":
        # Prepare training data with battery IDs
        print("\nPreparing training data...")
        result = prepare_data(TRAIN_BATTERIES)
        if result is None:
            return
        gas_data, capacity_hist, capacity_future, battery_ids = result
        
        # Use 5-fold cross-validation with completely random splitting
        k = 5
        print(f"\nStarting {k}-fold cross-validation with completely random splitting...")
        
        # Initialize fold statistics list
        fold_stats_list = []
        
        # Create a list of all sample indices
        all_indices = list(range(len(gas_data)))
        
        # Shuffle all indices randomly
        np.random.shuffle(all_indices)
        
        # Split into k folds
        fold_size = len(all_indices) // k
        global_folds = [[] for _ in range(k)]
        
        for fold in range(k):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < k - 1 else len(all_indices)
            global_folds[fold] = all_indices[start_idx:end_idx]
        
        # Perform k-fold cross-validation
        all_metrics = []
        
        for fold in range(k):
            print(f"\n--- Fold {fold+1}/{k} ---")
            
            # Get indices for this fold
            val_indices = global_folds[fold]
            # Get remaining indices for training
            train_indices = []
            for f in range(k):
                if f != fold:
                    train_indices.extend(global_folds[f])
            
            # Create training and validation sets
            train_gas = [gas_data[i] for i in train_indices]
            train_hist = [capacity_hist[i] for i in train_indices]
            train_future = [capacity_future[i] for i in train_indices]
            
            val_gas = [gas_data[i] for i in val_indices]
            val_hist = [capacity_hist[i] for i in val_indices]
            val_future = [capacity_future[i] for i in val_indices]
            
            # Print fold statistics
            print(f"  Train samples: {len(train_gas)}, Val samples: {len(val_gas)}")
            
            # Train the model on this fold
            fold_model, fold_time, fold_epochs = train_model(train_gas, train_hist, train_future, val_gas, val_hist, val_future, fold=fold+1)
            # Store fold statistics
            fold_stats = {
                'fold': fold+1,
                'time': fold_time,
                'epochs': fold_epochs
            }
            fold_stats_list.append(fold_stats)
        
        # Print summary of all folds
        print(f"\n--- 5-fold Cross-validation Completed --\n")
        print("Fold Training Summary:")
        print("Fold | Time (s) | Epochs")
        print("------------------------")
        total_time = 0
        total_epochs = 0
        for stats in fold_stats_list:
            print(f"{stats['fold']:4d} | {stats['time']:8.2f} | {stats['epochs']:6d}")
            total_time += stats['time']
            total_epochs += stats['epochs']
        print("------------------------")
        print(f"Avg  | {total_time/5:8.2f} | {total_epochs/5:6.1f}")
        print(f"Total| {total_time:8.2f} | {total_epochs:6d}")
        
    elif MODE == "predict":
        # Check if fold models exist for ensemble prediction
        fold_models = []
        for fold in range(1, 6):
            fold_path = os.path.join("predictors", f"hybrid_predictor_fold_{fold}.pth") if USE_REAL_GAS_DATA else os.path.join("predictors", f"compared_predictor_fold_{fold}.pth")
            if os.path.exists(fold_path):
                fold_models.append(load_model(fold_path))
            else:
                break
        
        if fold_models:
            print(f"\nLoaded {len(fold_models)} fold models for ensemble prediction with uncertainty quantification")
            validate_with_comparison(fold_models, TEST_BATTERY, is_ensemble=True)
        else:
            # Fallback to single model if no fold models exist
            hybrid_model = load_model(MODEL_PATH)
            validate_with_comparison(hybrid_model, TEST_BATTERY, is_ensemble=False)
    
    else:
        print(f"Invalid MODE: {MODE}. Please set MODE to either 'train' or 'predict' in the configuration.")
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main()