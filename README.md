# 3D-Mesh-Normalization-Quantization-and-Error-Analysis---Assignment


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Assignment](https://img.shields.io/badge/Assignment-Complete-brightgreen)


A complete implementation of 3D mesh preprocessing pipeline for AI applications, featuring normalization, quantization, error analysis, and advanced research components.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [License](#-license)

## 🎯 Overview

This project implements a complete 3D mesh preprocessing pipeline as per the company’s AI preprocessing assignment.
It focuses on normalization, quantization, reconstruction, and error analysis — essential steps before training AI models like SeamGPT.

## ✨ Features

### 🎓 Main Assignment 
- **Task 1**: Mesh loading, inspection, and statistical analysis
- **Task 2**: Two normalization methods (Min-Max, Unit Sphere) + Quantization (1024 bins)
- **Task 3**: Reconstruction with error measurement (MSE, MAE) and visualization

### 🚀 Bonus Task - Option 2
- **Rotation & Translation Invariance**: PCA-based normalization
- **Adaptive Quantization**: Dynamic bin allocation based on local geometric density
- **Comprehensive Analysis**: Uniform vs adaptive quantization comparison

## 📁 Project Structure

```bash
3d_mesh_assignment/
│
├── 📊 main.py                 # Main assignment tasks (1–3)
├── 🔧 mesh_processor.py       # Core mesh processing
├── 🚀 bonus_processor.py      # Bonus task implementation
├── 🎯 run_assignment.py       # Complete runner (main + bonus)
├── 📋 requirements.txt        # Dependencies
├── 📖 README.md               # Project documentation
│
├── input_meshes/
│   ├── branch.obj
│   ├── cylinder.obj
│   ├── explosive.obj
│   ├── fence.obj
│   ├── girl.obj
│   ├── person.obj
│   ├── table.obj
│   ├── talwar.obj
│
├── 📂 output/                 # Generated outputs
│   ├── 📂 normalized/         # Normalized meshes
│   ├── 📂 quantized/          # Quantized meshes
│   ├── 📂 reconstructed/      # Reconstructed meshes
│   ├── 📂 plots/              # Visualization plots
│   ├── 📂 bonus/              # Bonus task results
│   └── 📄 summary_report.txt  # Comprehensive analysis

```

## ⚙️ Technical Details

This project implements **3D Mesh Normalization, Quantization, and Error Analysis**, along with a **Bonus Task** for advanced reconstruction or optimization.  
It processes `.obj` mesh files by:
- Normalizing the meshes to a standard coordinate space  
- Quantizing vertex coordinates for efficient representation  
- Reconstructing the meshes from quantized data  
- Computing error metrics between original and reconstructed meshes  
- Visualizing and saving the results  

Developed using **Python 3.10+**, leveraging libraries like:
- `numpy` — numerical computations  
- `trimesh` — mesh loading and manipulation  
- `matplotlib` — visualization  
- `scipy` — geometric and mathematical utilities  

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sujithradevi03/3D-Mesh-Normalization-Quantization-and-Error-Analysis---Assignment.git
   cd 3D-Mesh-Normalization-Quantization-and-Error-Analysis---Assignment
   ```

2. Install dependencies
pip install -r requirements.txt

3. Run the main script (Main + Bonus):
python run_assignment.py
   
## 🚀 Usage

Place all our input .obj files inside the input_meshes/ folder.
(Already included: branch.obj, cylinder.obj, explosive.obj, fence.obj, girl.obj, person.obj, table.obj, talwar.obj)

Run the complete assignment (main + bonus) using:

python run_assignment.py


we can also run individual modules:

python main.py
python bonus_processor.py


Processed results and visualizations will be automatically saved in the output/ directory (once generated).

## 📊 Results

After running the project, we’ll obtain:

1. Normalized Meshes – in /output/normalized/

2. Quantized Meshes – in /output/quantized/

3. Reconstructed Meshes – in /output/reconstructed/

4. Error Plots and Metrics – in /output/plots/

5. Summary Report – detailed analysis in summary_report.txt

## 📄 License

This project is provided for academic and evaluation purposes.
© 2025 Sujithra Devi M — All Rights Reserved.
