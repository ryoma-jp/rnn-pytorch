
import os
from pathlib import Path
import torch
from torchinfo import summary
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class RNN(torch.nn.Module):
    """
    Recurrent Neural Network (RNN) for Time Series Analysis.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Constructor.

        Args:
            input_size (int): input size.
            hidden_size (int): hidden size.
            output_size (int): output size.
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward propagation.

        Args:
            x (torch.Tensor): input tensor.
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.linear(out[:, -1, :])
        return out
    
def train(X_train, y_train, n_epochs=1000, lr=0.001, save_path=None):
    """
    Train the RNN model.

    Args:
        X_train (torch.Tensor): training data.
        y_train (torch.Tensor): training label.
        n_epochs (int): number of epochs.
        lr (float): learning rate.
        save_path (str): path to save the trained model.
    """

    if (save_path is not None):
        os.makedirs(save_path, exist_ok=True)

    # --- Prepare the training data ---
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # --- Build the model ---
    model = RNN(input_size=1, hidden_size=64, output_size=1)
    summary(model)

    # --- Train the model ---
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            train_loss.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch: {epoch+1:04d}, train loss: {train_loss[-1]:.8f}')

    # --- Visualize the training loss ---
    if (save_path is not None):
        df_train_loss = pd.DataFrame({'train_loss': train_loss})
        sns.lineplot(data=df_train_loss)
        plt.savefig(os.path.join(save_path, 'train_loss.png'))
        plt.close()

    # --- save the trained model ---
    if (save_path is not None):
        torch.save(model.state_dict(), Path(save_path, 'rnn.pth'))

    return model

def predict(model, X_test):
    """
    Predict the test data.

    Args:
        model (torch.nn.Module): trained model.
        X_test (torch.Tensor): test data.
    """
    # --- Prepare the test data ---
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # --- Predict the test data ---
    y_pred = model(X_test)

    return y_pred.detach().numpy()

def evaluate(y_true, y_pred):
    """
    Evaluate the model.

    Args:
        y_true (torch.Tensor): true label.
        y_pred (torch.Tensor): predicted label.
    """
    # --- Calculate the MSE ---
    mse = mean_squared_error(y_true, y_pred)

    return mse
