# 🫐 GRAPES — GReylevel Analysis of ParticlES

GRAPES is a Python toolkit for quantitative analysis of grey-level intensities in X-ray tomograms of particles.
It was originally developed for studying core–shell behavior, internal cracking, and void formation within individual particles extracted from tomographic datasets.

### 📄 Reference / Methodology:
For detailed example use cases and citation, see the paper:
👉 https://doi.org/10.26434/chemrxiv-2024-msbc1

### 🚀 Features
- Automated particle property extraction into a Pandas DataFrame

- Radial analysis of grey-level intensities using the GREAT2 method

- Batch processing of large particle datasets

- Utility functions for plotting, file I/O, and visualization

- Designed for high-throughput analysis of tomographic data

### 🧠 Core Concept
At the heart of GRAPES is the radial layer analysis:
Each particle is divided into concentric layers, and properties such as mean grey-level intensity, standard deviation, and radial gradients are computed.
This enables quantitative comparison of features like shell thickness, internal cracks, and material heterogeneity across thousands of particles.

### 📦 Repository Structure
```bash
GRAPES/
│
├── GRAPES.py          # Core analysis functions and utilities
├── example_data/      # (Optional) Example particle datasets
├── examples/          # Example scripts and workflows
├── README.md          # Project overview (this file)
└── requirements.txt   # Python dependencies
```

### ⚙️ Installation
```bash
git clone https://github.com/YourUsername/GRAPES.git
cd GRAPES
pip install -r requirements.txt
```

### 🧾 License
See the [MIT License](LICENSE)
