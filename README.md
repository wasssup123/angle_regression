# MNIST Rotated Digit Angle Prediction

This project trains a CNN to predict the rotation angle of a rotated MNIST digit.

## Data Preprocessing

The data preprocessing steps are as follows:

1.  **Load MNIST Data:** The original MNIST training and testing datasets are loaded.
2.  **Create Subsets:** 5,000 random samples are selected from both the training and testing sets.
3.  **Rotate Digits:** A custom `RotatedDigits` dataset is created. This dataset rotates each digit by a random angle between -45 and 45 degrees.
4.  **Save Images:** The rotated digit images are saved to the `./use_data/train` and `./use_data/test` directories. The filename of each image contains the rotation angle.

To run the data preprocessing, execute the first cell in the `new_test.ipynb` notebook.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

1.  `nn.Conv2d(1, 8, 3, padding=1)`
2.  `nn.BatchNorm2d(8)`
3.  `nn.ReLU()`
4.  `nn.MaxPool2d(2,2)`
5.  `nn.Conv2d(8, 16, 3, padding=1)`
6.  `nn.BatchNorm2d(16)`
7.  `nn.ReLU()`
8.  `nn.MaxPool2d(2,2)`
9.  `nn.Conv2d(16, 32, 3, padding=1)`
10. `nn.BatchNorm2d(32)`
11. `nn.ReLU()`
12. `nn.Conv2d(32, 32, 3, padding=1)`
13. `nn.BatchNorm2d(32)`
14. `nn.ReLU()`
15. `nn.Flatten()`
16. `nn.Linear(32 * 7 * 7, 1)`

The number of nodes from the last convolutional layer to the MLP layer is `32 * 7 * 7 = 1568`.

## Training

To train the model, execute the second cell in the `new_test.ipynb` notebook. The model is trained for 100 epochs using the Mean Squared Error (MSE) loss function and the SGD optimizer. The best model is saved to `best_model.pt`.

## Performance

The model's performance is measured by the Root Mean Squared Error (RMSE) on the validation set. The best validation RMSE achieved is approximately **10.56**.
