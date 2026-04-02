# PyTorch LSTM Capacity Predictor for Battery Data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
warnings.filterwarnings('ignore')

# Configuration settings
CACHE_ROOT = ".\dataset_cache"
TRAIN_BATTERIES = ["11_1", "14", "15", "16"]  # Batteries used for training
TEST_BATTERY = "13"  # Battery used for testing (non-training battery)
DEFAULT_FRAG_LEN = 32  # Default input length (number of cycles)
DEFAULT_PRED_LEN = 32  # Default prediction length (number of cycles)
EPOCHS = 500  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimizer
PATIENCE = 50  # Early stopping patience
MIN_DELTA = 1e-6  # Minimum improvement required to reset patience

# Mode selection: "train" or "predict"
# MODE = "train"
MODE = "predict"  # Set to "predict" to run prediction mode

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_battery_data(battery_id):
    """Load battery data from npz file"""
    npz_path = os.path.join(CACHE_ROOT, f"cell{battery_id}.npz")
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found!")
        return None
    
    try:
        data = np.load(npz_path)
        # Extract cycle and QDischarge data
        cycle = data["cycle"]
        qdischarge = data["QDischarge"]/100
        
        # Create dataframe
        df = pd.DataFrame({
            "cycle": cycle,
            "QDischarge": qdischarge
        })
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"Loaded data for battery {battery_id}: {len(df)} cycles")
        return df
    except Exception as e:
        print(f"Error loading data for battery {battery_id}: {str(e)}")
        return None

def prepare_data(all_data, frag_len=DEFAULT_FRAG_LEN, pred_len=DEFAULT_PRED_LEN):
    """Prepare data for LSTM model training"""
    # Prepare data for each battery separately, then combine
    all_X = []
    all_y = []
    all_combined_qd = []
    
    # First collect all qd values for scaling reference
    for battery_id, df in all_data.items():
        qd_values = df["QDischarge"].values
        all_combined_qd.extend(qd_values)
    
    all_combined_qd = np.array(all_combined_qd)
    print(f"Total data points across all batteries: {all_combined_qd.shape}")
    
    # Normalize the data with fixed range based on battery physics
    # Use a fixed range of 0 to 1.1 to handle realistic battery capacity variations
    # This ensures that even if predictions go below the training data range (0.7), they can still be properly normalized
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Set fixed min and max values based on battery capacity physics
    # 0.0 represents a completely dead battery
    # 1.1 represents a new battery with slight overcapacity (common in new batteries)
    fixed_min = 0.5
    fixed_max = 1.1
    
    # Fit scaler with fixed min and max instead of using actual data range
    scaler.fit([[fixed_min], [fixed_max]])
    
    # Create input-output pairs for each battery separately
    for battery_id, df in all_data.items():
        qd_values = df["QDischarge"].values
        print(f"Processing battery {battery_id}: {len(qd_values)} cycles")
        
        # Normalize this battery's data
        scaled_qd = scaler.transform(qd_values.reshape(-1, 1))
        
        # Create input-output pairs for this battery
        for i in range(len(scaled_qd) - frag_len - pred_len + 1):
            X_segment = scaled_qd[i:i+frag_len, 0]
            y_segment = scaled_qd[i+frag_len:i+frag_len+pred_len, 0]
            all_X.append(X_segment)
            all_y.append(y_segment)
    
    # Convert lists to numpy arrays
    X = np.array(all_X)
    y = np.array(all_y)
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Shape: [samples, seq_len, features]
    y = torch.tensor(y, dtype=torch.float32)  # Shape: [samples, pred_len]
    
    print(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, scaler

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

def build_lstm_model(input_size=1, hidden_size=256, num_layers=4, output_size=DEFAULT_PRED_LEN):
    """Build LSTM model"""
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.to(device)
    print(model)
    return model

def train_model(model, train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, patience=PATIENCE, min_delta=MIN_DELTA):
    """Train LSTM model with early stopping"""
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    train_loss_history = []
    val_loss_history = []

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("\nTraining LSTM model...")
    print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    # Record training start time
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)

        # Print epoch results
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  Val loss improved to {val_loss:.6f}, resetting patience counter")
        else:
            patience_counter += 1
            if patience_counter % 10 == 0:
                print(f"  Val loss not improving, patience counter: {patience_counter}/{patience}")

        # Check if early stopping condition is met
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    # Calculate total training time
    end_time = time.time()
    total_training_time = end_time - start_time

    # Print training summary
    print(f"\nTraining completed in {total_training_time:.2f} seconds")
    print(f"Total epochs trained: {epoch + 1}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Load best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_loss_history, val_loss_history

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
    input_tensor = torch.tensor(input_segment_scaled, dtype=torch.float32).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()
    
    # Inverse transform to get actual values
    prediction = scaler.inverse_transform(prediction_scaled)
    prediction = prediction.flatten()
    
    # Apply physical constraints based on historical degradation patterns
    for i in range(len(prediction)):
        # Ensure capacity is within physically possible range
        prediction[i] = max(prediction[i], 0.0)
        prediction[i] = min(prediction[i], 1.1)
    
    return prediction

def plot_training_history(train_loss, val_loss):
    """Plot training history"""
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_training_history.png')
    plt.close()
    print("Training history plot saved as lstm_training_history.png")

def load_test_battery_cycles(battery_id):
    """Load individual cycle files for the test battery"""
    cycles_dir = os.path.join(CACHE_ROOT, f"cell{battery_id}_cycles")
    if not os.path.exists(cycles_dir):
        print(f"Error: Cycles directory {cycles_dir} not found!")
        return []

    # Get all cycle files
    cycle_files = [f for f in os.listdir(cycles_dir) if f.endswith('.npz') and f.startswith('cycle_')]
    print(f"Found {len(cycle_files)} cycle files for battery {battery_id}")

    # Extract cycle numbers from filenames
    cycle_numbers = []
    for file in cycle_files:
        # Extract cycle number from filename like "cycle_008.npz"
        try:
            cycle_num = int(file.split('_')[1].split('.')[0])
            cycle_numbers.append(cycle_num)
        except Exception as e:
            print(f"Error parsing cycle number from file {file}: {str(e)}")

    # Sort cycle numbers
    cycle_numbers.sort()
    print(f"Cycle numbers: {cycle_numbers}")
    return cycle_numbers

def load_battery_qdischarge(battery_id):
    """Load QDischarge data from the battery's overall npz file"""
    npz_path = os.path.join(CACHE_ROOT, f"cell{battery_id}.npz")
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found!")
        return None, None

    try:
        data = np.load(npz_path)
        cycle = data["cycle"]
        qdischarge = data["QDischarge"]
        
        # Divide by 100 if it's a percentage
        qdischarge = qdischarge / 100.0
        
        print(f"Loaded QDischarge data for battery {battery_id}: {len(cycle)} cycles")
        return cycle, qdischarge
    except Exception as e:
        print(f"Error loading QDischarge data for battery {battery_id}: {str(e)}")
        return None, None

def main(frag_len=DEFAULT_FRAG_LEN, pred_len=DEFAULT_PRED_LEN):
    """Main function to train and evaluate LSTM model"""
    
    if MODE == "train":
        print("Starting PyTorch LSTM capacity prediction model training...")
        print(f"Configuration: frag_len={frag_len}, pred_len={pred_len}")
        print(f"Training batteries: {TRAIN_BATTERIES}")
        
        # Load data for all training batteries
        all_data = {}
        for battery_id in TRAIN_BATTERIES:
            df = load_battery_data(battery_id)
            if df is not None:
                all_data[battery_id] = df
        
        if not all_data:
            print("Error: No valid data loaded!")
            return
        
        # Prepare data for training
        X, y, scaler = prepare_data(all_data, frag_len, pred_len)
        
        if X.shape[0] == 0:
            print("Error: Not enough data to create input-output pairs!")
            return
        
        # Split data into train, validation, and test sets
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        test_size = len(X) - train_size - val_size
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Build LSTM model
        model = build_lstm_model(output_size=pred_len)
        
        # Train model
        train_loss, val_loss = train_model(model, train_loader, val_loader)
        
        # Plot training history
        plot_training_history(train_loss, val_loss)
        
        # Create predictors folder if it doesn't exist
        os.makedirs("predictors", exist_ok=True)
        
        # Save the model
        model_path = os.path.join("predictors", "lstm_predictor.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved as {model_path}")
        
    elif MODE == "predict":
        print("Starting LSTM capacity prediction mode...")
        print(f"Configuration: frag_len={frag_len}, pred_len={pred_len}")
        print(f"Test battery: {TEST_BATTERY}")
        
        # Load test battery's individual cycle files to get cycle numbers
        cycle_numbers = load_test_battery_cycles(TEST_BATTERY)
        if not cycle_numbers:
            print("Error: No cycle files found for test battery!")
            return
        
        # Load test battery's QDischarge data
        cycle_array, qdischarge_array = load_battery_qdischarge(TEST_BATTERY)
        if cycle_array is None or qdischarge_array is None:
            print("Error: Failed to load QDischarge data!")
            return
        
        # Load training data to get scaler
        print("\nLoading training data to prepare scaler...")
        all_data = {}
        for battery_id in TRAIN_BATTERIES:
            df = load_battery_data(battery_id)
            if df is not None:
                all_data[battery_id] = df
        
        if not all_data:
            print("Error: No valid training data loaded for scaler!")
            return
        
        # Prepare data to get scaler (no need for full training)
        X_train, y_train, scaler = prepare_data(all_data, frag_len, pred_len)
        
        # Load trained model
        print("\nLoading trained model...")
        model = build_lstm_model(output_size=pred_len)
        try:
            model_path = os.path.join("predictors", "lstm_predictor.pth")
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Model loaded successfully from {model_path}!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please run the script in 'train' mode first to train the model.")
            return
        
        # Process each cycle number
        print("\nProcessing each cycle file...")
        
        # Create result folder named after TEST_BATTERY
        result_folder = os.path.join("LSTM_baseline", f"cell{TEST_BATTERY}_result")
        os.makedirs(result_folder, exist_ok=True)
        print(f"Results will be saved to folder: {result_folder}")
        
        for cycle_num in cycle_numbers:
            print(f"\n=== Processing Cycle {cycle_num} ===")
            
            # Find the index of this cycle in the cycle array
            cycle_idx = np.where(cycle_array == cycle_num)[0]
            if len(cycle_idx) == 0:
                print(f"Cycle {cycle_num} not found in QDischarge data!")
                continue
            
            cycle_idx = cycle_idx[0]
            print(f"Found cycle {cycle_num} at index {cycle_idx}")
            
            # Check if we have enough history data
            if cycle_idx < frag_len - 1:
                print(f"Not enough history data for cycle {cycle_num}: need {frag_len} cycles, but only have {cycle_idx + 1} cycles")
                continue
            
            # Extract input segment: [cycle_idx - frag_len + 1 : cycle_idx + 1]
            input_start_idx = cycle_idx - frag_len + 1
            input_segment = qdischarge_array[input_start_idx : cycle_idx + 1]
            print(f"Input segment: cycles {cycle_array[input_start_idx]} to {cycle_array[cycle_idx]}, shape: {input_segment.shape}")
            
            # Get actual future values if available
            actual_future = None
            if cycle_idx + 1 < len(qdischarge_array):
                actual_future = qdischarge_array[cycle_idx + 1 : ]
                target_length = len(actual_future)
                print(f"Actual future values available for {target_length} cycles")
            
            if actual_future is not None and target_length > 0:
                # Rolling prediction: predict until we reach target_length
                print("\n=== Starting Rolling Prediction ===")
                
                # Initialize input and predictions
                all_predictions = []
                current_input = input_segment.copy()
                
                # Keep predicting until we reach or exceed the target length
                while len(all_predictions) < target_length:
                    # Predict next chunk (always predict DEFAULT_PRED_LEN cycles)
                    pred_chunk = predict_segment(model, current_input, scaler, pred_len=DEFAULT_PRED_LEN)
                    
                    # Calculate how many predictions we need to add this iteration
                    remaining = target_length - len(all_predictions)
                    take_count = min(DEFAULT_PRED_LEN, remaining)
                    
                    # Add only the needed predictions to the list
                    all_predictions.extend(pred_chunk[:take_count])
                    print(f"  Predicted {take_count} cycles, total predictions so far: {len(all_predictions)}/{target_length}")
                    
                    # If we've reached the target length, break
                    if len(all_predictions) >= target_length:
                        break
                    
                    # Update input with FULL DEFAULT_PRED_LEN predictions for next iteration
                    # This maintains consistent rolling prediction mechanism
                    current_input = np.concatenate([current_input[DEFAULT_PRED_LEN:], pred_chunk])
                    print(f"  Updated input: shifted by {DEFAULT_PRED_LEN} cycles")
                
                # Plot actual vs predicted with rolling predictions
                plt.figure(figsize=(12, 6))
                plt.plot(range(frag_len), input_segment, label="Input History", color="blue")
                plt.plot(range(frag_len, frag_len + len(actual_future)), actual_future, label="Actual Future", color="green")
                plt.plot(range(frag_len, frag_len + len(all_predictions)), all_predictions, label="Predicted Future (Rolling)", color="red", linestyle="--")
                plt.axvline(x=frag_len - 1, color="black", linestyle="--", label="Current Cycle")
                plt.title(f"Rolling Capacity Prediction for Battery {TEST_BATTERY} - Cycle {cycle_num}")
                plt.xlabel(f"Cycle (relative to cycle {cycle_num})")
                plt.ylabel("Normalized QDischarge")
                plt.legend()
                plt.grid(True)
                
                # Save plot to result folder with unified naming and three-digit cycle number
                plot_filename = f"prediction_cycle_{cycle_num:03d}.png"
                plot_path = os.path.join(result_folder, plot_filename)
                plt.savefig(plot_path)
                plt.close()
                print(f"Prediction plot saved as {plot_path}")
            else:
                # Single prediction when no actual future data is available
                predictions = predict_segment(model, input_segment, scaler, pred_len)
                print(f"Predicted {pred_len} cycles")
                
                # Plot only prediction without actual values
                plt.figure(figsize=(12, 6))
                plt.plot(range(frag_len), input_segment, label="Input History", color="blue")
                plt.plot(range(frag_len, frag_len + pred_len), predictions, label="Predicted Future", color="red", linestyle="--")
                plt.axvline(x=frag_len - 1, color="black", linestyle="--", label="Current Cycle")
                plt.title(f"Capacity Prediction for Battery {TEST_BATTERY} - Cycle {cycle_num}")
                plt.xlabel(f"Cycle (relative to cycle {cycle_num})")
                plt.ylabel("Normalized QDischarge")
                plt.legend()
                plt.grid(True)
                
                # Save plot to result folder with unified naming and three-digit cycle number
                plot_filename = f"prediction_cycle_{cycle_num:03d}.png"
                plot_path = os.path.join(result_folder, plot_filename)
                plt.savefig(plot_path)
                plt.close()
                print(f"Prediction plot saved as {plot_path}")
        
        print("\nPrediction mode completed!")
        print(f"Generated prediction plots for {len(cycle_numbers)} cycles")
    
    else:
        print(f"Error: Invalid MODE '{MODE}'. Must be 'train' or 'predict'")

if __name__ == "__main__":
    # Run with default parameters
    main()
