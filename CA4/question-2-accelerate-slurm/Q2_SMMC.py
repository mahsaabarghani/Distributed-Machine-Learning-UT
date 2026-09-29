#1-???
import time 
start_time = time.time()
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from accelerate import Accelerator

accelerator = Accelerator()
test_x = np.load('./test_data/test_x.npy').astype(np.float32) 
test_y = np.load('./test_data/test_y.npy').astype(np.float32) 
test_dataset = TensorDataset(torch.tensor(test_x).float(), torch.tensor(test_y).long())

train_x = np.load("./train_data/train_x.npy").astype(np.float32)
train_y = np.load("./train_data/train_y.npy").astype(np.float32)
train_dataset = TensorDataset(torch.tensor(train_x).float(), torch.tensor(train_y).long())

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out) 
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.bn3(out)  
        out = self.relu3(out)
        out = self.fc4(out)
        return out

input_dim = train_x.shape[1]
hidden_dim = 64
output_dim = len(torch.unique(torch.tensor(train_y)))

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

iter = 0
accuracy = 0

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.requires_grad_()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        


        iter += 1
        
        if iter % 500 == 0:
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.requires_grad_()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            
            accuracy = 100 * correct / total
            
            print(f"Iteration: {iter}. Loss: {loss.item():.4f}. Accuracy: {accuracy:.2f}%")



if accuracy >= 80:
    print("accuracy >= 80")
else:
    print("accuracy < 80")
#1-?
torch.save(model.state_dict(), "model_checkpoint.pth")
print("model saved as a checkpoint.")

end_time = time.time()

print(f"Total Time: {end_time - start_time:.2f} seconds")
