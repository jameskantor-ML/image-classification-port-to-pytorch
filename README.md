# Food-101 Image Classification: TensorFlow to PyTorch Migration

A complete reimplementation of a food image classification system, migrating from TensorFlow to PyTorch while maintaining academic rigor and adding production-ready improvements.

## Project Overview

This project ports a high-performing image classification CNN originally developed as part of a Boston University Master's program group project. The original implementation achieved strong academic results using TensorFlow on Google Colab with A100 GPU compute. Post-graduation, the need to run locally on an RTX 5090 GPU—which lacked TensorFlow driver support at the time—motivated a complete framework migration to PyTorch.

### Dataset
- **Food-101**: 101,000 images across 101 food categories
- Hosted on Hugging Face Datasets
- Stratified 80/10/10 train/validation/test split
- ~1,000 images per class

### Model Architecture
- **Base Model**: MobileNetV2 with ImageNet pretraining
- **Transfer Learning**: Fine-tuned entire network with exponential learning rate decay
- **Input Resolution**: 160×160 RGB images
- **Optimization**: Mixed precision training (FP16) with gradient scaling

## Key Features

### Core Functionality
- Complete PyTorch implementation of transfer learning pipeline
- Stratified dataset splitting maintaining class balance
- Custom PyTorch Dataset class with on-demand image loading
- Mixed precision training for improved performance
- Early stopping with model checkpointing
- Comprehensive evaluation metrics (Top-1, Top-5 accuracy, confusion matrix)

### Enhanced Features Added During Migration
1. **Data Quality Pipeline**
   - Automated corrupted image detection
   - Pre-training data cleaning (identified and removed 3/101,000 corrupted images)
   - Class distribution validation and visualization

2. **Production Readiness**
   - Deprecation warning suppressors for cleaner logs
   - Memory-efficient DataLoader configuration
   - Proper error handling for edge cases

3. **Evaluation Tools**
   - Class balance verification across splits
   - Top-k accuracy calculation (k=1,5)
   - Normalized confusion matrix visualization
   - Per-class performance analysis

## Results

### Model Performance
- **Validation Accuracy**: 71.35% (Epoch 6)
- **Training Accuracy**: 83.28% (Epoch 7)
- **Test Metrics**: Top-1 and Top-5 accuracy on held-out test set

### Training Characteristics
- Consistent learning trajectory with healthy convergence
- Minimal overfitting (12% train-val gap at convergence)
- Exponential learning rate decay (5e-5 → 4.4e-5)
- Early stopping triggered appropriately

## Technical Stack

### Frameworks & Libraries
```python
- PyTorch 2.x
- torchvision
- Hugging Face Datasets
- scikit-learn (metrics, splitting)
- NumPy, Pandas
- Matplotlib, Seaborn
```

### Hardware Requirements
- CUDA-compatible GPU (tested on RTX 5090)
- 8GB+ VRAM recommended
- Mixed precision training enabled

## Project Structure

```
├── Final_Project_Working_Version_CURRENT.ipynb  # Main notebook
├── README.md
└── requirements.txt
```

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/image-classification-port-to-pytorch.git
cd image-classification-port-to-pytorch

# Install dependencies
pip install torch torchvision datasets scikit-learn seaborn tqdm

# Launch Jupyter
jupyter notebook Final_Project_Working_Version_CURRENT.ipynb
```

## Usage

The notebook is organized into sequential sections:

1. **Data Loading & Preprocessing**: Loads Food-101, removes corrupted images, creates stratified splits
2. **Model Definition**: MobileNetV2 classifier with transfer learning setup
3. **Training Pipeline**: Custom training loop with mixed precision and early stopping
4. **Evaluation**: Comprehensive metrics including confusion matrix and top-k accuracy

Simply run cells sequentially. Training takes approximately 1-2 hours on an RTX 5090.

## Important Notes

### Markdown Discrepancies
The notebook markdown contains references to the original TensorFlow implementation's results. Some stated results may differ from the PyTorch execution outputs. The primary goal was ensuring feature parity and functional correctness across frameworks, not exact result replication (which is expected due to framework differences in initialization, numerical precision, and data augmentation).

### Academic Context
This represents a post-graduation enhancement of coursework. The original ML training wrapper from the BU course remains largely intact, demonstrating that well-designed abstractions can transcend framework boundaries.

## Key Learnings

### Framework Migration Insights
- PyTorch's dynamic computation graph simplified debugging compared to TensorFlow
- Mixed precision training required explicit scaler management in PyTorch
- DataLoader configuration critical for GPU memory management
- Hugging Face Datasets integration seamless across both frameworks

### Best Practices Implemented
- Stratified splitting to maintain class balance
- Validation set for hyperparameter tuning, test set for final evaluation only
- Data quality checks before training (corruption detection)
- Comprehensive logging and visualization

## Future Improvements

To reach 80-90% validation accuracy:
- Increase input resolution to 224×224
- Enhanced data augmentation (rotation, color jitter, affine transforms)
- Progressive unfreezing training strategy
- Larger architecture (EfficientNet-B3, ResNet50)
- Ensemble methods with multiple architectures

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Boston University Master's Program for original project foundation
- Food-101 dataset creators
- Hugging Face for dataset hosting
- Course instructors for ML pipeline design patterns

## Contact

For questions or collaboration: [Your contact information]

---

**Note**: This is a personal educational project demonstrating framework migration skills and ML engineering best practices. Not intended for production deployment without further validation.
