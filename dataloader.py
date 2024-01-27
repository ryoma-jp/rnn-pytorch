
import statsmodels.datasets.co2 as co2
import numpy as np

class DataLoader_Statsmodels_CO2():
    """
    Load and analyze atmospheric CO2 data.
    """

    def __init__(self):
        """
        Load Atmospheric CO2 data and return the data as a pandas DataFrame.
        """
        df_co2 = co2.load().data
        print(f'<< df_co2.head() >>\n{df_co2.head()}\n')

        # --- Check the data ---
        print(f'<< df_co2.describe() >>\n{df_co2.describe()}\n')
        print(f'<< df_co2.isnull().sum() >>\n{df_co2.isnull().sum()}\n')
        
        # --- Complement the missing value ---
        self.df_co2 = df_co2.interpolate()

        # --- Normalization ---
        self.df_co2_min = self.df_co2['co2'].min()
        self.df_co2_max = self.df_co2['co2'].max()
        self.df_co2 = (self.df_co2['co2'] - self.df_co2_min) / (self.df_co2_max - self.df_co2_min)

        return None
        
    def split_data(self):
        """
        Split the data into training data and test data.
        """
        # --- Split the data into training data and test data ---
        df_co2_train = self.df_co2[:'1989-12-31']
        df_co2_test = self.df_co2['1990-01-01':]

        return df_co2_train, df_co2_test
    
    def preprocessing(self, df_co2, window_size=16, n_samples=1):
        """
        Preprocessing the data.

        Args:
            df_co2 (pandas.DataFrame): dataset.
            window_size (int): window size.
            n_samples (int): number of samples.
        """
        # --- Pre-processing ---
        n_data = len(df_co2)-window_size-1
        X = np.zeros([n_data, window_size, 1])
        y = np.zeros([n_data, 1])

        for t in range(n_data):
            X[t] = df_co2.values[t:t+window_size].reshape(-1, 1)
            y[t] = df_co2.values[t+window_size]

        return X, y
    
    def inverse_transform(self, y_pred):
        """
        Inverse transform the predicted value.

        Args:
            y_pred (numpy.ndarray): predicted value.
        """
        y_pred = y_pred * (self.df_co2_max - self.df_co2_min) + self.df_co2_min
        return y_pred
    