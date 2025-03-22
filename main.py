import pandas as pd
import torch.nn as nn
import torch
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = '/Users/suryapasupuleti/Documents/coding/AI coding/heart_disease/heart.csv'
df = pd.read_csv(path)


# ** PREPROCESS DATA **

# features cols
features = df.loc[:, :"thal"]
# target cols
target = df.loc[:, "target"]

#mean of each column is places if null
df.fillna(df.mean, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=30, test_size=0.2)

print(x_train)

# Scale Data
sc = StandardScaler()
train_data = sc.fit_transform(x_train) # mean to 0 and sd to 1
test_data = sc.transform(x_test) # apply the same on to this

x_train = torch.from_numpy(x_train.to_numpy().astype(np.float32))
x_test = torch.from_numpy(x_test.to_numpy().astype(np.float32))

y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32))
y_test = torch.from_numpy(y_test.to_numpy().astype(np.float32))




y_train = y_train.view(y_train.shape[0], 1)  # Ensure it's (n_samples, 1)
y_test = y_test.view(y_test.shape[0], 1) 



# ** Model **

'''
class RandomForest(nn.Module):
    def __init__(self, x):
        super(RandomForest, x).__init__()
        forest = nn.Random  
'''

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        
        # First hidden layer (64 neurons)
        self.fc1 = nn.Linear(input_dim, 64)  # First layer: input_dim -> 64
        self.relu1 = nn.ReLU()  # ReLU activation function
        
        # Second hidden layer (32 neurons)
        self.fc2 = nn.Linear(64, 32)  # Second layer: 64 -> 32
        self.relu2 = nn.ReLU()  # ReLU activation function
        
        # Third hidden layer (16 neurons)
        self.fc3 = nn.Linear(32, 16)  # Third layer: 32 -> 16
        self.relu3 = nn.ReLU()  # ReLU activation function
        
        # Output layer (1 neuron for binary classification)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary output
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    
n_features = x_train.shape[1]
learning_rate = 0.0001

model = LinearRegression(n_features)

criterion = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10000

for epoch in range(num_epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)

    optimiser.zero_grad()
    loss.backward()

    optimiser.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch + 1}, Loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x_test)
    y_pred_results = y_predicted.round()
    acc = y_pred_results.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')







