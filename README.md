# Pytorch-Image-classification-models
Deep learning model for classification in pytorch framework

## Data
### CIFAR 10
- `input_dim`: (32, 32)
- `output_dim`: 10

### CIFAR 100
- `input_dim`: (32, 32)
- `output_dim`: 100

### ImageNet
- `input_dim`: (224, 224)
- `output_dim`: 1000

### MNIST
- `input_dim`: (224, 224)
- `output_dim`: 10

### Fasion-MNIST
- `input_dim`: (224, 224)
- `output_dim`: 10

### SVHN
- `input_dim`: (32, 32)
- `output_dim`: 10

### STL 10
- `input_dim`: (96,96)
- `output_dim`: 10

## Model
### CoAtNet
- `CoAtNet-0`
- `CoAtNet-1`
- `CoAtNet-2`
- `CoAtNet-3`
- `CoAtNet-4`

### DenseNet
- `DenseNet-121`
- `DenseNet-169`
- `DenseNet-201`
- `DenseNet-264`

### GoogLeNet

### Inception
- `Inception-v3`

### ResNet
- `ResNet-18`
- `ResNet-34`
- `ResNet-50`
- `ResNet-101`
- `ResNet-152`

### Vision Transformer
- Vision Transformer 1D
  - `ViT-Base`
  - `ViT-Large`
  - `ViT-Huge`
- Vision Transformer 2D
  - `ViT-Base`
  - `ViT-Large`
  - `ViT-Huge`
- Vision Transformer 3D

## Save File
### Save Path
```python
check_path = f"../trained_models/{args.current_date}_{args.modelname}_{args.dataset}_{args.comment}"
```

### Save Loss & Accuracy File
- `log_train.csv`
- `log_valid.csv`
- `log_test.csv`

### Save Code
- `main.py`

### Save Checkpoint
