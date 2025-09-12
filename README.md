Markdown

# Industrial HMI-based Anomaly Detection System

## Project Overview

This project develops a machine learning-based anomaly detection system specifically designed for Industrial Control Systems (ICS) environments. It addresses the limitations of traditional, static security measures by implementing a dynamic, data-driven approach to identify subtle cyber-physical threats and operational anomalies. The core of the system is an unsupervised **Isolation Forest** model, which learns the "normal" behavior of an industrial process and flags deviations. This intelligence is then presented to operators via an intuitive **Human-Machine Interface (HMI)** for real-time monitoring and actionable insights.

The system was developed and validated using the **WaDi (Water Distribution) dataset**, simulating a realistic industrial setting with both normal operations and known attack scenarios.

## Features

* **Unsupervised Anomaly Detection:** Utilizes an Isolation Forest model that effectively identifies anomalies without requiring pre-labeled attack data.
* **Real-time Simulation:** Processes data points in a simulated real-time stream for dynamic monitoring.
* **Intuitive HMI:** A Tkinter-based graphical user interface provides:
    * Live plotting of multiple sensor values using Matplotlib.
    * Visual markers (red dots) for detected anomalies on graphs.
    * A dynamic event log displaying timestamps, anomaly scores, and root cause predictions.
    * Control buttons (Start, Pause, Stop) for simulation management.
* **Root Cause Analysis:** Pinpoints the specific sensor most responsible for a detected anomaly, providing actionable intelligence.
* **Data Preprocessing Pipeline:** Includes robust methods for handling missing values, data type conversion, and normalization.
* **Model Persistence:** Saves and loads the trained model for efficient real-time deployment.
* **IEC 62443 Alignment:** Designed with principles of industrial cybersecurity standards for Integrity (SR 3.2), Event Logging (SR 5.2), and Availability (SR 7.1).

## Project Structure

.
├── src/
│   ├── preprocess_data.py          # Data loading, cleaning, normalization
│   ├── train_model.py              # Isolation Forest model training and persistence
│   ├── real_time_prediction_test.py # HMI implementation, real-time simulation, anomaly detection, root cause analysis
│   └── utils/                      # (Optional) Helper functions/scripts
├── data/
│   ├── WADI_1B.csv                 # Raw WaDi dataset (or relevant portion)
│   ├── WADI_test_dataset.csv       # Test dataset with simulated anomalies
│   └── isolation_forest_model.joblib # Saved trained ML model
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── (other project documentation/reports)


## Setup and Installation

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    (Remember to replace `your-username/your-repo-name` with your actual GitHub details.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Data:** Ensure your `WADI_1B.csv` (for training) and `WADI_test_dataset.csv` (for testing/HMI) are placed in the `data/` directory.

## Usage

### 1. Data Preprocessing

Run the `preprocess_data.py` script to clean and normalize the raw WaDi dataset. This script generates the processed data that will be used for training.

```bash
python src/preprocess_data.py
2. Model Training
Train the Isolation Forest model using the preprocessed data. This will save the trained model as isolation_forest_model.joblib.

Bash

python src/train_model.py
3. Run Real-time HMI Simulation
Execute the main application script to launch the HMI, which will simulate real-time data flow, detect anomalies, and display results.

Bash

python src/real_time_prediction_test.py
Once the HMI window appears:

Click the "Start" button to begin the real-time simulation and anomaly detection.

Observe the live sensor data graph and the anomaly log.

Red markers on the graph and "Anomaly" entries in the log indicate detected deviations.

Use "Pause" and "Stop" to control the simulation.

Core Technologies
Python 3.x

Pandas: Data acquisition and manipulation.

Scikit-learn: Machine learning (Isolation Forest, MinMaxScaler).

Joblib: Model persistence.

Tkinter: Graphical User Interface (HMI) development.

Matplotlib: Real-time data visualization and plotting.

NumPy: Numerical operations.

Contributing
(Optional - you can add details here if you plan to accept contributions)

License
This project is licensed under the MIT License - see the LICENSE file for details.
(You might need to create a https://www.google.com/search?q=LICENSE file if you don't have one.)

Acknowledgments
The WaDi dataset creators (Chaudhry et al.) for providing a valuable research testbed.

REVA University for support and guidance.
