# SCICanvas Installation Guide

## Prerequisites

- Python 3.10 (recommended)
- CUDA-compatible GPU (recommended for deep learning models)
- At least 8GB RAM (16GB+ recommended for large datasets)

## Installation Methods

### Method 1: Conda Environment (Recommended)

The easiest way to install SCICanvas is using the provided conda environment file:

```bash
# Clone or navigate to the SCICanvas directory
cd SCICanvas

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate scicanvas

# Install SCICanvas in development mode
pip install -e .
```

### Method 2: Manual Conda Setup

```bash
# Create a new conda environment
conda create -n scicanvas python=3.10
conda activate scicanvas

# Install PyTorch with CUDA support
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install SCICanvas
pip install -e .
```

### Method 3: Pip Only Installation

```bash
# Create virtual environment
python -m venv scicanvas_env
source scicanvas_env/bin/activate  # On Windows: scicanvas_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SCICanvas
pip install -e .
```

## Dependency Categories

### Core Dependencies (Verified Working Versions)
- **PyTorch**: 2.5.1 with CUDA 11.8 support
- **NumPy**: 2.0.0 for numerical computing
- **SciPy**: 1.14.1 for scientific computing
- **Pandas**: 2.2.3 for data manipulation
- **Scikit-learn**: 1.5.2 for machine learning utilities

### Single-Cell Analysis
- **Scanpy**: 1.11.1 for single-cell analysis
- **AnnData**: 0.11.4 for annotated data structures
- **UMAP**: 0.5.7 for dimensionality reduction
- **PyNNDescent**: 0.5.13 for nearest neighbor search

### Visualization
- **Matplotlib**: 3.10.0 for plotting
- **Seaborn**: 0.13.2 for statistical visualization
- **Pillow**: 11.0.0 for image processing
- **OpenCV**: 4.10.0.84 for computer vision

### Machine Learning & Experiment Tracking
- **TensorBoard**: 2.19.0 for experiment tracking
- **Weights & Biases**: 0.19.11 for experiment management

### Development Tools
- **Jupyter**: 1.1.1 for interactive development
- **JupyterLab**: 4.3.4 for advanced notebook interface

## Verification

After installation, verify that SCICanvas works correctly:

```bash
# Activate environment
conda activate scicanvas

# Test basic import
python -c "import scicanvas; print('✓ SCICanvas imported successfully')"

# Run validation script
python validate_structure.py

# Test examples (optional)
python examples/single_cell_example.py
python examples/protein_example.py
python examples/materials_example.py
```

## Environment Reproduction

For exact environment reproduction, you can use the exported environment file:

```bash
# Create environment from exact export
conda create --name scicanvas --file cv_environment.txt
```

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CPU-only version if needed
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Memory Issues
For large datasets or limited memory:
- Reduce batch sizes in examples
- Use CPU instead of GPU for smaller models
- Close other applications to free memory

### Package Conflicts
If you encounter package conflicts:
```bash
# Clean conda cache
conda clean --all

# Create fresh environment
conda env remove -n scicanvas
conda env create -f environment.yml
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended Requirements
- Python 3.10
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 10GB disk space

## Notes

- The requirements.txt and environment.yml files are generated from the working CV conda environment
- All package versions have been tested and verified to work together
- For production use, consider pinning all dependency versions
- Some packages may have additional system dependencies (e.g., CUDA drivers)

## Support

If you encounter installation issues:
1. Check that your system meets the requirements
2. Ensure CUDA drivers are properly installed (for GPU support)
3. Try creating a fresh conda environment
4. Refer to the individual package documentation for specific issues 