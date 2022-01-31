import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class Data():
    def __init__(self) -> None:
        self.x_train = None
        self.x_test =  None
        self.y_test =  None
        self.y_train = None
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        self.read_data()
        
    def read_data(self):

        print(f'Shape of train data is: {self.train_data.shape}\nShape of test data is: {self.test_data.shape}')

        self.x_train, self.y_train = self.train_data.iloc[:, :-2], self.train_data.iloc[:, -1:]
        self.x_test, self.y_test = self.test_data.iloc[:, :-2], self.test_data.iloc[:, -1:]
        self.x_test, self.y_test = self.test_data.iloc[:, :-2], self.test_data.iloc[:, -1:]
        scaling_data = MinMaxScaler() 
        le = LabelEncoder()
        self.y_train = le.fit_transform(self.y_train)
        self.y_test = le.fit_transform(self.y_test)
        self.x_train = scaling_data.fit_transform(self.x_train)
        self.x_test = scaling_data.transform(self.x_test)
