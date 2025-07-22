# 🛡️ One<sup>2</sup>: An Intrusion Detection System for Both Internal and External Vehicular Network from Weak Labeled Data
A PyTorch implementation of One<sup>2</sup> for intrusion detection across multiple IVN and EVN datasets.

## 📁 Project Structure

```
├── main.py                 # Main entry point
├── data_processor.py       # Data preprocessing and graph construction  
├── models.py              # HGT model definition
├── trainer.py             # Training and evaluation
├── utils.py               # Utility functions
├── dataset/               # Dataset files
├── model/                 # Saved checkpoints
├── output/                # Output files
```

## 🛠️ Environment Setup

### Create Conda Environment

```bash
# Create a new conda environment
conda create -n hgt-ids python=3.8 -y

# Activate the environment
conda activate hgt-ids
```

### Install from requirements.txt

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Sample requirements.txt:**
```
torch>=1.12.0
torch-geometric>=2.3.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
tensorboard>=2.10.0
tqdm>=4.64.0
```

## 🚀 Run Commands

### Basic Training

```bash
# Train model on CIC-IDS2017 dataset (binary classification)
python main.py --dataset CIC-IDS2017 --binary True --epochs 20
```

### Advanced Configuration

```bash
# Custom hyperparameters
python main.py \
    --dataset TON_IoT \
    --file df.csv \
    --binary True \
    --batchsize 128 \
    --epochs 100 \
    --learning_rate 0.01 \
    --hidden_channels 64 \
    --num_heads 2 \
    --num_layers 1 \
    --cuda cuda:0

# Evaluation only (no training)
python main.py --dataset TON_IoT --train_if False --roc True

# Enable ROC curve generation
python main.py --dataset CIC-IDS2017 --roc True --epochs 50
```

### Supported Datasets

| Dataset | Command |
|---------|---------|
| CIC-IDS2017 | `--dataset CIC-IDS2017` |
| TON_IoT | `--dataset TON_IoT` |
| CAR-HACKING | `--dataset CAR-HACKING` |
| CAN-intrusion | `--dataset CAN-intrusion` |
| CIC-UNSW-NB15 | `--dataset CIC-UNSW-NB15` |
| CICIoV2024 | `--dataset CICIoV2024` |

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `TON_IoT` | Experimental dataset |
| `--file` | str | `df.csv` | Dataset file name |
| `--binary` | bool | `True` | Binary or multiclass classification |
| `--batchsize` | int | `128` | Batch size for training |
| `--epochs` | int | `1` | Number of training epochs |
| `--learning_rate` | float | `0.01` | Learning rate |
| `--hidden_channels` | int | `64` | Hidden layer dimensions |
| `--num_heads` | int | `2` | Number of attention heads |
| `--num_layers` | int | `1` | Number of HGT layers |
| `--cuda` | str | `cuda:0` | CUDA device |
| `--train_if` | bool | `True` | Whether to train the model |
| `--roc` | bool | `True` | Generate ROC curves |
| `--attack` | bool | `False` | Enable adversarial evaluation |

### Monitor Training

```bash
# Start TensorBoard to monitor training progress
tensorboard --logdir=logs/
```

### Multi-GPU Training

The code supports multi-GPU training. Set the GPUs in the script:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --dataset TON_IoT --epochs 100
```

---

## 📊 Results and Outputs

- **Model checkpoints**: Saved in `./model/` directory
- **Training metrics**: CSV files in `./output/` directory  
- **ROC curves**: Data saved in `./roc/` directory
- **TensorBoard logs**: Available in `./logs/` directory

## 🤝 Contributing

Feel free to submit issues and pull requests to improve this implementation!

## 📄 License

This project is open source. Please cite appropriately if used in research.
