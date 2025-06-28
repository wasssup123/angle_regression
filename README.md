# MNIST Rotation Angle Regression (MATLAB Version)

This project re-implements the MATLAB official example of a CNN regression model in PyTorch to predict the rotation angle of MNIST digit images.

## Project Structure

- `train_matlab.ipynb`: Main training notebook  
- `DigitsDataTrain.mat`: MATLAB-format training data  
- `DigitsDataTest.mat`: MATLAB-format test data  
- `best_model.pt`: Saved best model checkpoint (see **Model Checkpoint** below)  

## Dataset

We use MATLAB `.mat` files for the MNIST rotation dataset:  
- **Training set**: `DigitsDataTrain.mat`  
- **Test set**: `DigitsDataTest.mat`  
- **Validation set**: 15% split from the training set  

## Model Architecture

The `EnhancedCNNRegressor` CNN model:

```text
Input: 1×28×28 grayscale image
│
├─ Conv2d(1→8,   3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2×2)
├─ Conv2d(8→16,  3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2×2)
├─ Conv2d(16→32, 3×3, pad=1) → BatchNorm → ReLU
├─ Conv2d(32→32, 3×3, pad=1) → BatchNorm → ReLU
├─ Flatten
└─ Linear(32×7×7 → 1) → Output rotation angle
```

## Training Parameters

- **Batch size**: 128  
- **Optimizer**: SGD (lr=1e-3, momentum=0.9)  
- **LR scheduler**: StepLR (step_size=20, γ=0.1)  
- **Epochs**: 100  
- **Loss**: Mean Squared Error (MSE)  
- **Metric**: Root Mean Squared Error (RMSE)  

## Usage

1. Install dependencies:
   ```bash
   pip install torch numpy scipy matplotlib
   ```
2. Place `DigitsDataTrain.mat` and `DigitsDataTest.mat` in the project root.  
3. Open and run all cells in `train_matlab.ipynb`.

## Training Output

- Each epoch prints training and validation RMSE.  
- When validation RMSE improves, the model weights are saved.  
- After training, RMSE curves are plotted.

## Model Checkpoint

- **Filename**: `best_model.pt`  
- **Location**: Project root (`/Users/longheishe/Documents/minst_regression/best_model.pt`)  
- **Contents**: PyTorch `state_dict` of the `EnhancedCNNRegressor` with the lowest validation RMSE.  
- **Load checkpoint**:
  ```python
  model = EnhancedCNNRegressor().to(device)
  model.load_state_dict(torch.load("best_model.pt"))
  model.eval()
  ```

## Notes

- Runs on MPS (Mac GPU) if available, otherwise CPU.  
- Uses `scipy.io` to load MATLAB `.mat` files.  
- Validation split uses a fixed random seed for reproducibility.

## output 
Best at epoch 23: Val RMSE=5.2134,  Best validation RMSE=5.2134 at epoch 23
