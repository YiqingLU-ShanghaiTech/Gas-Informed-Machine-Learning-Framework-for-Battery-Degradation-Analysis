import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Default parameters
Gas_start = 70  # Gas data start position (each gas has 200 points: 10 zeros + 180 data + 10 zeros)
Gas_end = 120   # Gas data end position (each gas has 200 points: 10 zeros + 180 data + 10 zeros)
# Default PCA components per gas
DEFAULT_PCA_COMPONENTS = {
    0: 3,  # CO
    1: 5,  # CO2
    2: 10, # C2H4
    3: 1,  # CH4
    4: 1   # C2H6
}

# Font settings for plotting
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Set fonts for normal display
plt.rcParams['axes.unicode_minus'] = False    # Display negative signs correctly

class GasDataKMeansAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.all_data = []
        self.cell_ids = []
        self.sheet_names = []
        self.sheet_indices = []
        self.processed_data = None
        self.train_indices = None
        self.train_data = None
        self._use_training_data_only = False  # Flag to control whether to use only training data in preprocessing
        self.pca_components = None  # PCA components per gas
        self.pca = None  # PCA model
        self.scaler = None  # StandardScaler model
        self.kmeans = None  # Kmeans model

    def load_data(self):
        """Load data from all sheets in the Excel file"""
        print(f"Loading data from: {self.file_path}")
        # Reset data storage lists
        self.all_data = []
        self.cell_ids = []
        self.sheet_indices = []
        excel_file = pd.ExcelFile(self.file_path)
        self.sheet_names = excel_file.sheet_names
        
        print(f"Found {len(self.sheet_names)} sheets: {', '.join(self.sheet_names)}")
        
        for sheet_idx, sheet_name in enumerate(self.sheet_names):
            # Read data from each sheet
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            print(f"  Sheet '{sheet_name}' has {len(df)} rows")
            
            # Extract feature data (exclude cycle number and label, use only 600 gas sampling points)
            features = df.iloc[:, 1:-1].values  # Exclude first column (cycle number) and last column (label)
            print(f"  Extracted {len(features)} feature rows with shape {features.shape}")
            
            # Convert numpy array to list and extend
            self.all_data.extend(features.tolist())
            
            # Record cell ID and cycle number for each data point
            cycles = df.iloc[:, 0]
            print(f"  Found {len(cycles)} cycles in sheet '{sheet_name}'")
            
            for cycle in cycles:
                self.cell_ids.append(f"{sheet_name}_cycle_{int(cycle)}")
                self.sheet_indices.append(sheet_idx)  # Record which sheet each data point belongs to
        
        self.all_data = np.array(self.all_data)
        print(f"\nSummary after loading:")
        print(f"Successfully loaded {len(self.all_data)} data records")
        print(f"Feature dimension: {self.all_data.shape[1]}")
        print(f"Total cell_ids created: {len(self.cell_ids)}")
        print(f"Total sheet_indices recorded: {len(self.sheet_indices)}")
        
        # Verify data consistency
        if len(self.all_data) != len(self.cell_ids) or len(self.all_data) != len(self.sheet_indices):
            print("WARNING: DATA INCONSISTENCY DETECTED!")
            print(f"  all_data length: {len(self.all_data)}")
            print(f"  cell_ids length: {len(self.cell_ids)}")
            print(f"  sheet_indices length: {len(self.sheet_indices)}")
        
        return self

    def split_data_by_sheet(self, fixed_train_sheets=['cell11_1', 'cell14', 'cell15', 'cell16']):
        """Split data into training and test sets based on sheet names
        
        Args:
            fixed_train_sheets: List of sheet names to be used as training set
            
        Returns:
            self
        """
        # Get indices of data points from specified training sheets
        train_indices = []
        for i, sheet_idx in enumerate(self.sheet_indices):
            if self.sheet_names[sheet_idx] in fixed_train_sheets:
                train_indices.append(i)
        
        # Store indices and create training data slice
        self.train_indices = np.array(train_indices, dtype=int)
        
        if len(self.train_indices) > 0:
            self.train_data = self.all_data[self.train_indices]
            print(f"Selected training data: {len(self.train_indices)} samples from {len(fixed_train_sheets)} sheets")
        else:
            raise ValueError(f"No data found for training sheets: {', '.join(fixed_train_sheets)}")
        
        return self

    def preprocess_data(self, gas_start=Gas_start, gas_end=Gas_end, gas_selection=None, use_global_var=True):
        """Data preprocessing: select specified range of points for selected gases
        
        Args:
            gas_start: Start position in each gas's 200-point data
            gas_end: End position in each gas's 200-point data
            gas_selection: List of gas indices to use (0=CO, 1=CO2, 2=C2H4, 3=CH4, 4=C2H6), default is [0, 1, 2]
            use_global_var: Whether to use global variables
            
        Returns:
            Processed data (when use_global_var=False) or self (when use_global_var=True)
        """
        # Use complete dataset by default, only use training data when explicitly requested
        use_training_only = getattr(self, '_use_training_data_only', False)
        data_source = self.train_data if (use_training_only and hasattr(self, 'train_data')) else self.all_data
        
        # Set parameters and gas selection
        params = {'gas_start': (Gas_start if use_global_var else gas_start),
                  'gas_end': (Gas_end if use_global_var else gas_end)}
        gas_selection = gas_selection or [0, 1, 2]
        
        # Get actual number of points from data to ensure dimension match
        gas_base_idx = 0
        start_idx = gas_base_idx + (params['gas_start'] - 1)
        end_idx = gas_base_idx + params['gas_end']
        num_points = len(data_source[0, start_idx:end_idx])
        
        # Initialize processed data array
        total_features = num_points * len(gas_selection)
        processed_data = np.zeros((len(data_source), total_features), dtype=np.float64)
        
        # Process each sample and selected gases
        for i in range(len(data_source)):
            feature_idx = 0
            for gas_idx in gas_selection:
                # Calculate indices (gas_start is 1-indexed, convert to 0-indexed)
                gas_base_idx = gas_idx * 200
                start_idx = gas_base_idx + (params['gas_start'] - 1)
                end_idx = gas_base_idx + params['gas_end']
                
                # Extract and assign gas data
                gas_data = data_source[i, start_idx:end_idx]
                processed_data[i, feature_idx:feature_idx+len(gas_data)] = gas_data
                feature_idx += len(gas_data)
        
        # Store processed data
        self.processed_data = processed_data
        self.selected_gases = gas_selection
        
        return self if use_global_var else processed_data

    def apply_pca(self, pca_components):
        """Apply PCA dimensionality reduction
        
        Args:
            pca_components: List of PCA components to keep for each selected gas
            
        Returns:
            self
        """
        # Calculate total PCA components
        total_components = sum(pca_components)
        
        # Normalize data
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(self.processed_data)
        
        # Apply PCA
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=total_components, random_state=42)
        self.pca_data = self.pca.fit_transform(normalized_data)
        
        # Store PCA components
        self.pca_components = pca_components
        
        # Gas names for display
        gas_names = ['CO', 'CO2', 'C2H4', 'CH4', 'C2H6']
        selected_gas_names = [gas_names[i] for i in self.selected_gases]
        
        print(f"PCA applied with {total_components} components (" 
              f"per gas: {dict(zip(selected_gas_names, pca_components))})")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return self

    def perform_kmeans(self, n_clusters):
        """Perform K-means clustering
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (labels, kmeans_model)
        """
        # Validate number of clusters
        if n_clusters < 2:
            raise ValueError("Number of clusters must be greater than or equal to 2")
        
        print(f"Performing K-means clustering (K={n_clusters})...")
        print(f"Clustering using processed dataset, dataset size: {self.processed_data.shape}")
        
        # Use PCA transformed data if available, otherwise use processed data
        if hasattr(self, 'pca_data') and self.pca_data is not None:
            print(f"Using PCA-transformed data for clustering, shape: {self.pca_data.shape}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.pca_data)
        else:
            # Normalize data for better clustering results
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(self.processed_data)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(normalized_data)
        
        # Count data volume in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_stats = dict(zip(unique, counts))
        print("Distribution of data across clusters:")
        for cluster, count in cluster_stats.items():
            print(f"Cluster {cluster}: {count} samples ({count/len(labels)*100:.2f}%)")
        
        # Calculate various evaluation metrics
        try:
            if hasattr(self, 'pca_data') and self.pca_data is not None:
                silhouette_avg = silhouette_score(self.pca_data, labels)
                ch_score = calinski_harabasz_score(self.pca_data, labels)
                db_score = davies_bouldin_score(self.pca_data, labels)
            else:
                normalized_data = scaler.transform(self.processed_data)
                silhouette_avg = silhouette_score(normalized_data, labels)
                ch_score = calinski_harabasz_score(normalized_data, labels)
                db_score = davies_bouldin_score(normalized_data, labels)
            print(f"Silhouette coefficient: {silhouette_avg:.4f}")
            print(f"Calinski-Harabasz index: {ch_score:.4f}")
            print(f"Davies-Bouldin index: {db_score:.4f}")
        except Exception as e:
            print(f"Error calculating evaluation metrics: {str(e)}")
        
        # Return labels and kmeans model to match main function's expectation
        return labels, kmeans

    def visualize_clusters(self, labels, n_clusters):
        """Visualize clustering results (using PCA dimensionality reduction)"""
        try:
            from sklearn.decomposition import PCA
            
            print("Performing PCA dimensionality reduction for visualization...")
            
            # Ensure labels and data have the same length to prevent errors
            if len(labels) != len(self.processed_data):
                print("Warning: Labels and processed data have different lengths. Using first N samples.")
                min_len = min(len(labels), len(self.processed_data))
                processed_data_subset = self.processed_data[:min_len]
                labels_subset = labels[:min_len]
            else:
                processed_data_subset = self.processed_data
                labels_subset = labels
            
            # Use PCA to reduce high-dimensional data to 2D for visualization
            pca = PCA(n_components=2, random_state=42)
            reduced_data = pca.fit_transform(processed_data_subset)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                 c=labels_subset, cmap='viridis', 
                                 s=50, alpha=0.6, edgecolors='w', linewidths=0.5)
            plt.colorbar(scatter, label='Cluster Label')
            plt.title(f'K-means Clustering Results Visualization (K={n_clusters})')
            plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(f'kmeans_visualization_k{n_clusters}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"PCA dimensionality reduction preserved information: {explained_variance:.4f}")
            
        except ImportError:
            print("Failed to load PCA for visualization, please ensure scikit-learn is correctly installed")
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_results(self, labels, n_clusters):
        """Save clustering results"""
        # Get number of gases used
        num_gases = len(self.selected_gases) if hasattr(self, 'selected_gases') else 3
        
        # Create DataFrame with cell IDs, cluster labels, and gas count information
        results = pd.DataFrame({
            'cell_id_cycle': self.cell_ids[:len(labels)],
            'cluster_label': labels,
            'num_clusters': n_clusters,
            'num_gases': num_gases
        })
        
        # Add PCA components information if available
        if hasattr(self, 'pca_components') and self.pca_components is not None:
            # Gas names for display
            gas_names = ['CO', 'CO2', 'C2H4', 'CH4', 'C2H6']
            selected_gas_names = [gas_names[i] for i in self.selected_gases]
            pca_info = dict(zip(selected_gas_names, self.pca_components))
            results['pca_components'] = str(pca_info)
        
        # Save results to CSV file - append mode
        output_file = 'kmeans_clustering_results.csv'
        file_exists = os.path.isfile(output_file)
        
        # Save the file with append mode
        results.to_csv(output_file, index=False, mode='a' if file_exists else 'w', header=not file_exists)
        print(f"Clustering results for {num_gases} gases have been {'appended to' if file_exists else 'saved to'}: {output_file}")
        
        return results

def main():
    # File path
    file_path = 'datasets/NormalizedGasData.xlsx'
    
    # Create analyzer instance
    analyzer = GasDataKMeansAnalyzer(file_path)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Use fixed training sheets
        fixed_train_sheets = ['cell11_1', 'cell14', 'cell15', 'cell16']
        print(f"\nUsing fixed training sheets: {', '.join(fixed_train_sheets)}")
        
        # Split data to get training data indices
        analyzer.split_data_by_sheet(fixed_train_sheets=fixed_train_sheets)
        
        # Let user select gases
        print("\nAvailable gases:")
        print("0: CO")
        print("1: CO2")
        print("2: C2H4")
        print("3: CH4")
        print("4: C2H6")
        
        # Get gas selection from user
        gas_input = input("\nPlease enter gas indices to use (comma-separated, default: 0,1,2): ").strip()
        if gas_input:
            try:
                gas_selection = [int(g) for g in gas_input.split(',')]
                # Validate gas indices
                if all(0 <= g <= 4 for g in gas_selection):
                    print(f"Selected gases: {gas_selection}")
                else:
                    print("Invalid gas indices, using default [0,1,2]")
                    gas_selection = [0, 1, 2]
            except ValueError:
                print("Invalid input, using default [0,1,2]")
                gas_selection = [0, 1, 2]
        else:
            gas_selection = [0, 1, 2]
            print("Using default gases: CO, CO2, C2H4")
        
        # Get Gas_start from user
        default_gas_start = Gas_start
        gas_start_input = input(f"\nPlease enter Gas_start position (default: {default_gas_start}): ").strip()
        try:
            gas_start = int(gas_start_input) if gas_start_input else default_gas_start
            print(f"Selected Gas_start: {gas_start}")
        except ValueError:
            print(f"Invalid input, using default Gas_start: {default_gas_start}")
            gas_start = default_gas_start
        
        # Get Gas_end from user
        default_gas_end = Gas_end
        gas_end_input = input(f"\nPlease enter Gas_end position (default: {default_gas_end}): ").strip()
        try:
            gas_end = int(gas_end_input) if gas_end_input else default_gas_end
            print(f"Selected Gas_end: {gas_end}")
        except ValueError:
            print(f"Invalid input, using default Gas_end: {default_gas_end}")
            gas_end = default_gas_end
        
        # Get PCA components for each selected gas
        gas_names = ['CO', 'CO2', 'C2H4', 'CH4', 'C2H6']
        selected_gas_names = [gas_names[i] for i in gas_selection]
        pca_components = []
        
        print("\nPlease enter number of PCA components for each selected gas:")
        for i, gas_idx in enumerate(gas_selection):
            gas_name = selected_gas_names[i]
            default_components = DEFAULT_PCA_COMPONENTS[gas_idx]
            pca_input = input(f"  {gas_name}: (default: {default_components}): ").strip()
            try:
                components = int(pca_input) if pca_input else default_components
                pca_components.append(components)
            except ValueError:
                print(f"Invalid input for {gas_name}, using default {default_components}")
                pca_components.append(default_components)
        
        print(f"Selected PCA components: {dict(zip(selected_gas_names, pca_components))}")
        
        # Get number of clusters from user
        user_k = input("\nPlease enter number of clusters (K-value, default: 2): ").strip()
        try:
            n_clusters = int(user_k) if user_k else 2
            print(f"Selected number of clusters: {n_clusters}")
        except ValueError:
            print("Invalid input, using default number of clusters: 2")
            n_clusters = 2
        
        # Step 1: Preprocess and train on training data
        print("\nStep 1: Training model on selected training data...")
        
        # Set to use only training data for preprocessing
        analyzer._use_training_data_only = True
        
        # Preprocess training data
        analyzer.preprocess_data(gas_selection=gas_selection, gas_start=gas_start, gas_end=gas_end)
        
        # Apply PCA on training data
        analyzer.apply_pca(pca_components)
        
        # Train K-means on training data
        train_labels, analyzer.kmeans = analyzer.perform_kmeans(n_clusters)
        
        # Step 2: Apply trained model on all data
        print("\nStep 2: Applying trained model on all data...")
        
        # Set to use all data
        analyzer._use_training_data_only = False
        
        # Preprocess all data using the same parameters
        analyzer.preprocess_data(gas_selection=gas_selection, gas_start=gas_start, gas_end=gas_end)
        
        # Apply PCA transformation using the trained PCA model
        if analyzer.scaler is not None and analyzer.pca is not None:
            normalized_all_data = analyzer.scaler.transform(analyzer.processed_data)
            all_pca_data = analyzer.pca.transform(normalized_all_data)
            
            # Predict clusters for all data
            all_labels = analyzer.kmeans.predict(all_pca_data)
            
            # Visualize clustering results
            analyzer.visualize_clusters(all_labels, n_clusters)
            
            # Save results
            results = analyzer.save_results(all_labels, n_clusters)
            
            print("\nAnalysis completed!")
            print(f"Clustering results saved, you can view each sample's cluster in 'kmeans_clustering_results.csv'.")
            print(f"Visualization chart saved as 'kmeans_visualization_k{n_clusters}.png'.")
        else:
            print("Error: Model training failed, could not apply to all data")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
