import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from tkinter import scrolledtext
from PIL import Image, ImageTk
import os

# --- 1. CONFIGURATION ---
# Define the path to your WaDi dataset CSV file.
# IMPORTANT: Ensure this file is in the same directory as this script.
DATASET_FILE = 'WADI_gpu_test_dataset.csv'

# Define the path to your topology image file.
# IMPORTANT: Ensure this file is in the same directory as this script.
TOPOLOGY_IMAGE_FILE = 'Sample WaDi Plant Prototype.jpg'

# Define the columns you have selected for the project.
SELECTED_FEATURES = ['Motorised Valve Status', 'Pump Status', 'Level Switch Alarm', 'Leval Transmitter', 'Flow Transmitter', 'Analyser Transmitter']
# The label column for attacks, as specified by the user.
LABEL_COLUMN = 'Flow Switch Alarm'

# Define a mapping for feature names to their units for the legend.
FEATURE_UNITS = {
    'Motorised Valve Status': 'Status',
    'Pump Status': 'Status',
    'Level Switch Alarm': 'Alarm',
    'Leval Transmitter': 'Level (m)',
    'Flow Transmitter': 'Flow (m^3/hr)',
    'Analyser Transmitter': 'pH',
}

# --- 2. DATA LOADING & PREPROCESSING ---

def load_and_prepare_data(filepath, features, label_col):
    """
    Loads the dataset, selects specified columns, and separates normal from attack data.
    """
    try:
        # Load the dataset using pandas
        df = pd.read_csv(filepath)
        print("WaDi Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Creating a synthetic dataset for demonstration.")
        df = create_synthetic_data()
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None, None, None, None, None
        
    # Check if all selected columns and the label column exist in the DataFrame
    missing_cols = [col for col in features + [label_col] if col not in df.columns]
    if missing_cols:
        print(f"KeyError: The following columns were not found in the CSV file: {missing_cols}")
        print("Please check the column names in your CSV file and update the SELECTED_FEATURES and LABEL_COLUMN variables in the script.")
        return None, None, None, None, None

    # Drop rows with any missing values in the selected features
    df_clean = df.dropna(subset=features + [label_col]).copy()
    
    # Attempt to convert the label column to a numeric type to ensure filtering works correctly
    try:
        df_clean.loc[:, label_col] = pd.to_numeric(df_clean[label_col], errors='coerce')
        # Drop rows where the conversion failed
        df_clean = df_clean.dropna(subset=[label_col]).copy()
    except Exception as e:
        print(f"Warning: Could not convert '{label_col}' to a numeric type. Skipping conversion.")
    
    # Filter for training data (normal operation)
    # Assuming '0' indicates a normal state based on the dataset
    train_data = df_clean[df_clean[label_col] == 0].copy()
    
    # Check if there is any training data to avoid the ValueError
    if train_data.empty:
        print("ValueError: Training data is empty. The model cannot be trained.")
        print("This may be because there are no rows with a '0' in the '{}' column.".format(label_col))
        print(f"The unique values found in the '{label_col}' column were: {df_clean[label_col].unique()}")
        print("Please check your dataset to confirm that it contains normal data for training.")
        return None, None, None, None, None
    
    # The IsolationForest model will only be trained on normal data.
    X_train = train_data[features]
    
    # Calculate statistics for the original unscaled training data
    train_mean = X_train.mean()
    train_std = X_train.std()
    
    # Normalize the training data for better model performance
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use the full cleaned dataset for real-time simulation, including anomalies
    simulation_data = df_clean.copy()
    
    # Return the scaled training data, the full simulation data, the scaler, and the stats
    return X_train_scaled, simulation_data, scaler, train_mean, train_std

def create_synthetic_data():
    """Creates a small, synthetic dataset for testing purposes."""
    np.random.seed(42)
    data = {
        'Motorised Valve Status': np.random.randint(0, 2, 200),
        'Pump Status': np.random.randint(0, 2, 200),
        'Level Switch Alarm': np.random.randint(0, 2, 200),
        'Leval Transmitter': np.random.normal(75, 2, 200),
        'Flow Transmitter': np.random.normal(50, 5, 200),
        'Analyser Transmitter': np.random.normal(10, 1, 200),
        'Flow Switch Alarm': np.zeros(200, dtype=int)
    }
    
    # Introduce some anomalies (spikes, drops)
    data['Flow Transmitter'][100:105] = np.random.normal(10, 2, 5)
    data['Leval Transmitter'][150:155] = np.random.normal(90, 5, 5)
    data['Flow Switch Alarm'][100:105] = 1
    data['Flow Switch Alarm'][150:155] = 1

    return pd.DataFrame(data)

# --- 3. TRAINING THE ISOLATION FOREST MODEL ---

def train_model(X_train_scaled):
    """
    Trains the Isolation Forest model on the normalized training data.
    """
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train_scaled)
    print("Isolation Forest model trained successfully on normal data.")
    return model

# --- 4. HMI (GUI) IMPLEMENTATION ---

class AnomalyDetectionHMI:
    def __init__(self, master, model, data, features, scaler, train_mean, train_std):
        self.master = master
        self.model = model
        self.data = data
        self.features = features
        self.scaler = scaler
        self.current_index = 0
        self.is_running = False  # Initially paused
        self.after_id = None
        self.anomaly_data = []

        # Store normal data statistics for reason prediction
        self.train_mean = train_mean
        self.train_std = train_std
        self.WINDOW_SIZE = 100

        master.title("Industrial Anomaly Detection HMI")
        master.geometry("1400x800")
        master.configure(bg="#2c3e50")
        
        self.main_frame = tk.Frame(master, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # PanedWindow for resizable sections
        self.paned_window = tk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg="#2c3e50")
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Frame for the Normal Pattern Table (left side)
        self.normal_pattern_frame = tk.Frame(self.paned_window, bg="#2c3e50")
        self.paned_window.add(self.normal_pattern_frame, width=300)

        # Title for the table
        tk.Label(self.normal_pattern_frame, text="Normal Value Pattern", font=("Helvetica", 14, "bold"), fg="white", bg="#2c3e50").pack(pady=10)
        
        # Create and populate the normal pattern table
        self.create_normal_pattern_table()

        # Graph and Image Container Frame (right side of the paned window)
        self.graph_and_image_container = tk.Frame(self.paned_window, bg="#2c3e50")
        self.paned_window.add(self.graph_and_image_container, width=900)

        # Frame for the Matplotlib graph (left part of graph_and_image_container)
        self.graph_frame = tk.Frame(self.graph_and_image_container, bg="#2c3e50")
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(10, 5), facecolor="#34495e")
        self.ax.set_title("Real-time Sensor Data", color="white")
        self.ax.set_xlabel("Recent Time (Samples)", color="white")
        self.ax.set_ylabel("Normalized Value", color="white")
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_facecolor("#34495e")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.lines = {}
        for feature in self.features:
            unit = FEATURE_UNITS.get(feature, "Value")
            self.lines[feature], = self.ax.plot([], [], label=f"{feature} ({unit})")
        
        self.anomaly_markers = {feature: self.ax.plot([], [], 'rx', markersize=10)[0] for feature in self.features}
        
        self.ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, facecolor="#34495e", edgecolor='white', labelcolor='white')
        self.fig.tight_layout(rect=[0, 0.1, 1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Frame for the Topology Image (right part of graph_and_image_container)
        self.image_display_frame = tk.Frame(self.graph_and_image_container, bg="#2c3e50")
        self.image_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))

        # Add the "Plant Prototype" heading
        tk.Label(self.image_display_frame, text="Plant Prototype", font=("Helvetica", 14, "bold"), fg="white", bg="#2c3e50").pack(pady=(10, 5))

        # Load and display the image with an added file existence check
        self.tk_topology_image = None # Initialize to None
        if os.path.exists(TOPOLOGY_IMAGE_FILE):
            try:
                original_image = Image.open(TOPOLOGY_IMAGE_FILE)
                # We'll use a placeholder size for initial PhotoImage creation.
                # The image will be resized dynamically using configure_image_on_resize.
                self.topology_image = original_image
                self.tk_topology_image = ImageTk.PhotoImage(self.topology_image.resize((1,1), Image.LANCZOS)) 

                self.image_label = tk.Label(self.image_display_frame, image=self.tk_topology_image, bg="#2c3e50")
                self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                print(f"Topology image '{TOPOLOGY_IMAGE_FILE}' loaded successfully.")
                
                # Bind the configure_image_on_resize function to the image_display_frame's <Configure> event
                self.image_display_frame.bind('<Configure>', self.configure_image_on_resize)

            except Exception as e:
                print(f"An error occurred while loading the image: {e}")
                tk.Label(self.image_display_frame, text=f"Error loading image: {e}", fg="red", bg="#2c3e50", font=("Helvetica", 12)).pack()
        else:
            print(f"Error: The image file '{TOPOLOGY_IMAGE_FILE}' was not found. Please ensure it is in the same folder as the Python script.")
            tk.Label(self.image_display_frame, text="Image Not Found", fg="red", bg="#2c3e50", font=("Helvetica", 12)).pack()

        # Summary Frame with fixed height and a scrollbar
        self.summary_frame = tk.Frame(self.main_frame, bg="#2c3e50", height=150)
        self.summary_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.summary_frame.pack_propagate(False)

        # Headers for the summary table
        self.summary_headers_frame = tk.Frame(self.summary_frame, bg="#34495e")
        self.summary_headers_frame.pack(fill=tk.X)
        headers = ["Timestamp", "Status", "Anomaly Score", "Reason"] + list(self.features)
        self.tree = ttk.Treeview(self.summary_frame, columns=headers, show='headings')
        for col in headers:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree.tag_configure('anomaly', background='#e74c3c', foreground='white')
        self.tree.tag_configure('normal', background='#2ecc71', foreground='white')

        self.control_frame = tk.Frame(master, bg="#2c3e50")
        self.control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = tk.Label(self.control_frame, text="System Status: Paused", font=("Helvetica", 18, "bold"), fg="white", bg="#f39c12", padx=15, pady=10)
        self.status_label.pack(fill=tk.X)

        self.button_frame = tk.Frame(self.control_frame, bg="#2c3e50")
        self.button_frame.pack(pady=10)

        tk.Button(self.button_frame, text="Start", command=self.start_simulation, bg="#2ecc71", fg="white", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="Pause", command=self.pause_simulation, bg="#f39c12", fg="white", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="Stop", command=self.stop_simulation, bg="#e74c3c", fg="white", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="Restart", command=self.restart_simulation, bg="#3498db", fg="white", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        
        self.update_plot()
        
    def create_normal_pattern_table(self):
        """Creates a table displaying the normal mean and standard deviation for each feature."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#34495e", foreground="white", fieldbackground="#34495e", borderwidth=0)
        style.map("Treeview", background=[("selected", "#1a2c3a")])
        style.configure("Treeview.Heading", background="#2c3e50", foreground="white", font=("Helvetica", 10, "bold"), borderwidth=0)

        columns = ("Feature", "Mean", "Std Dev")
        self.pattern_tree = ttk.Treeview(self.normal_pattern_frame, columns=columns, show="headings", height=len(self.features))
        
        for col in columns:
            self.pattern_tree.heading(col, text=col)
            self.pattern_tree.column(col, width=100)
        
        for feature in self.features:
            mean_val = self.train_mean.get(feature, 0)
            std_val = self.train_std.get(feature, 0)
            self.pattern_tree.insert("", "end", values=(feature, f"{mean_val:.2f}", f"{std_val:.2f}"))
        
        self.pattern_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_summary(self, idx, original_values, score, status, reason="N/A"):
        tag = 'anomaly' if status == "Anomaly" else 'normal'
        data_to_insert = [str(idx), status, f"{score:.2f}", reason] + [f"{v:.2f}" for v in original_values]
        self.tree.insert('', 'end', values=data_to_insert, tags=(tag,))
        self.tree.yview_moveto(1)

    def stop_simulation(self):
        """Stops the simulation and resets all data."""
        self.is_running = False
        if self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None
            
        self.current_index = 0
        self.anomaly_data = []
        self.tree.delete(*self.tree.get_children())
        self.status_label.config(text="System Status: Stopped", bg="#e74c3c")
        for line in self.lines.values():
            line.set_data([], [])
        for markers in self.anomaly_markers.values():
            markers.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
    
    def restart_simulation(self):
        """Restarts the simulation from the beginning."""
        self.stop_simulation()
        self.is_running = True
        self.status_label.config(text="System Status: Running", bg="#2ecc71")
        self.update_plot()

    def start_simulation(self):
        """Starts or resumes the simulation."""
        if not self.is_running:
            self.is_running = True
            self.status_label.config(text="System Status: Running", bg="#2ecc71")
            self.update_plot()
            
    def pause_simulation(self):
        """Pauses the simulation."""
        self.is_running = False
        self.status_label.config(text="System Status: Paused", bg="#f39c12")
        if self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None
    
    def predict_anomaly_reason(self, current_data_point):
        """
        Analyzes an anomalous data point to find the most likely reason.
        The reason is the feature with the highest deviation from its normal mean.
        """
        deviation_scores = {}
        for feature in self.features:
            current_value = current_data_point[feature]
            normal_mean = self.train_mean.get(feature, 0)
            normal_std = self.train_std.get(feature, 0)
    
            # Calculate the absolute deviation in terms of standard deviations (Z-score)
            if normal_std > 0:
                deviation = abs(current_value - normal_mean) / normal_std
            else:
                deviation = abs(current_value - normal_mean) # Handle features with zero variance
    
            deviation_scores[feature] = deviation
    
        if not deviation_scores:
            return "Unknown", None, None
    
        most_deviated_feature = max(deviation_scores, key=deviation_scores.get)
        deviation_value = deviation_scores[most_deviated_feature]
    
        # Return a descriptive reason
        reason = f"High deviation in {most_deviated_feature} ({FEATURE_UNITS.get(most_deviated_feature, 'Value')})."
        return reason, most_deviated_feature, deviation_value

    def update_plot(self):
        if not self.is_running:
            return

        if self.current_index >= len(self.data):
            self.status_label.config(text="Simulation Complete", bg="#95a5a6")
            print("End of data stream. Simulation complete.")
            self.is_running = False
            return

        current_data_point = self.data.iloc[self.current_index]
        
        X_test_unscaled = pd.DataFrame([current_data_point[self.features]])
        X_test_scaled = self.scaler.transform(X_test_unscaled)
        
        prediction = self.model.predict(X_test_scaled)[0]
        anomaly_score = self.model.decision_function(X_test_scaled)[0]

        # Use a sliding window for plotting
        start_index = max(0, self.current_index - self.WINDOW_SIZE)
        data_to_plot = self.data.iloc[start_index:self.current_index + 1][self.features]
        data_to_plot_scaled = self.scaler.transform(data_to_plot)

        # Plot the data
        x_data = np.arange(start_index, self.current_index + 1)
        for i, feature in enumerate(self.features):
            y_data_scaled = data_to_plot_scaled[:, i]
            self.lines[feature].set_data(x_data, y_data_scaled)
        
        if prediction == -1:
            # Predict the reason for the anomaly
            anomaly_reason, _, _ = self.predict_anomaly_reason(current_data_point)
            self.anomaly_data.append(self.current_index)
            self.status_label.config(text=f"WARNING: ANOMALY DETECTED! Reason: {anomaly_reason}", bg="#e74c3c")
            print(f"Anomaly detected at index {self.current_index}. Score: {anomaly_score:.2f}. Reason: {anomaly_reason}")
            # Pass the reason to the summary table
            self.update_summary(self.current_index, current_data_point[self.features], anomaly_score, "Anomaly", anomaly_reason)
        else:
            self.status_label.config(text="System Status: Running", bg="#2ecc71")
            # Pass "N/A" for the reason to the summary table
            self.update_summary(self.current_index, current_data_point[self.features], anomaly_score, "Normal")
        
        # Update anomaly markers
        for i, feature in enumerate(self.features):
            anomaly_x_all = [idx for idx in self.anomaly_data if start_index <= idx <= self.current_index]
            
            y_coords = [data_to_plot_scaled[idx - start_index][i] for idx in anomaly_x_all]
            
            self.anomaly_markers[feature].set_data(anomaly_x_all, y_coords)
        
        # Set the x-axis limits to match the sliding window
        self.ax.set_xlim(start_index, start_index + self.WINDOW_SIZE)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
            
        self.current_index += 1
        
        if self.is_running:
            self.after_id = self.master.after(500, self.update_plot)

    def configure_image_on_resize(self, event):
        """
        Resizes the image to fit the image_display_frame while maintaining aspect ratio.
        This function is called whenever the image_display_frame changes size.
        """
        if self.topology_image: # Only proceed if an image was successfully loaded
            # Get the current width and height of the image_display_frame
            frame_width = event.width - 10 # Subtract padding
            frame_height = event.height - 40 # Subtract heading height and padding

            if frame_width <= 0 or frame_height <= 0:
                return # Avoid division by zero or invalid sizes

            # Calculate new dimensions while maintaining aspect ratio
            original_width, original_height = self.topology_image.size
            aspect_ratio = original_width / original_height

            if (frame_width / frame_height) > aspect_ratio:
                # Frame is wider than image, fit by height
                new_height = frame_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Frame is taller than image, fit by width
                new_width = frame_width
                new_height = int(new_width / aspect_ratio)

            # Ensure image doesn't become too small (optional, adjust as needed)
            if new_width < 50 or new_height < 50: 
                 return 
            
            resized_image = self.topology_image.resize((new_width, new_height), Image.LANCZOS)
            self.tk_topology_image = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=self.tk_topology_image)
            self.image_label.image = self.tk_topology_image # Keep a reference!


# --- 5. MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    X_train_scaled, simulation_data, data_scaler, train_mean, train_std = load_and_prepare_data(DATASET_FILE, SELECTED_FEATURES, LABEL_COLUMN)

    if X_train_scaled is not None and len(X_train_scaled) > 0:
        anomaly_model = train_model(X_train_scaled)

        root = tk.Tk()
        app = AnomalyDetectionHMI(root, anomaly_model, simulation_data, SELECTED_FEATURES, data_scaler, train_mean, train_std)
        root.mainloop()
    else:
        print("Could not proceed due to an error with the dataset.")