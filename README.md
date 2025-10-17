# 🫐 GRAPES — GReylevel Analysis of ParticlES

GRAPES is a Python toolkit for quantitative analysis of grey-level intensities in X-ray tomograms of particles.
It was originally developed for studying core–shell behavior, internal cracking, and void formation within individual particles extracted from tomographic datasets.

## Reference / Methodology:
For detailed example use cases, see the papers:

👉 [Demonstrating Faster Multi-Label Grey-Level Analysis for Crack Detection in Ex Situ and Operando Micro-CT Images of NMC Electrode](https://onlinelibrary.wiley.com/doi/full/10.1002/smtd.202500082)

👉 [Non-linear cracking response to voltage revealed by operando X-ray tomography in polycrystalline NMC811](https://pubs.rsc.org/en/content/articlelanding/2025/eb/d5eb00008d))

## Features
- Automated particle property extraction into a Pandas DataFrame

- Radial analysis of grey-level intensities using the GREAT2 method

- Batch processing of large particle datasets

- Utility functions for plotting, file I/O, and visualization

- Designed for high-throughput analysis of tomographic data

## Core Concept
At the heart of GRAPES is the radial layer analysis:
Each particle is divided into concentric layers, and properties such as mean grey-level intensity, standard deviation, and radial gradients are computed.
This enables quantitative comparison of features like shell thickness, internal cracks, and material heterogeneity across thousands of particles.

## Cloud Notebooks

You can explore GRAPES directly in your browser — no installation required!  
Click the badges below to launch the example notebooks on **Binder**.

| Example | Description | Binder Link |
|----------|--------------|--------------|
| **Example 1** | Micro-CT example | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MPJ-Imaging/GRAPES/HEAD?labpath=examples/example_1.ipynb) |
| **Example 2** | Nano-CT example | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MPJ-Imaging/GRAPES/HEAD?labpath=examples/example_2.ipynb) |

> 💡 **Tip:** Binder can take a minute or two to start — once loaded, the notebooks will open in JupyterLab with all dependencies preinstalled.


## Repository Structure
```bash
GRAPES/
│
├── GRAPES.py          # Core analysis functions and utilities
├── examples_data/      # (Optional) Example particle datasets
├── examples/          # Example scripts and workflows
├── README.md          # Project overview (this file)
└── requirements.txt   # Python dependencies
```

## Installation
```bash
git clone https://github.com/MPJ-Imaging/GRAPES.git
cd GRAPES
pip install -r requirements.txt
```

## License
See the [MIT License](LICENSE)
