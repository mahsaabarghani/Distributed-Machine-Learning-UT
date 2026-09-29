import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))

device = "cuda:0"
data_path = "/storage/dmls/stl10_data"
learning_rate = 0.001
training_epochs = 10
batch_size = 32

train_set = dsets.STL10(data_path, split="train", transform=transforms.ToTensor(), download=True)
test_set = dsets.STL10(data_path, split="test", transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)

class STL10CNN(nn.Module):
    def __init__(self):
        super(STL10CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(12 * 12 * 128, 256)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 10)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

model = STL10CNN().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

start_time = time.time()

for epoch in range(training_epochs):
    avg_cost = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        count += 1
        if count % 50 == 0:
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.max(outputs, 1)[1]
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            accuracy = correct * 100 / total
            loss_list.append(loss.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            print(f"Iteration: {count}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

end_time = time.time()
cuda_mem = torch.cuda.max_memory_allocated(device)
print(f"Training time: {end_time - start_time:.2f}s")
print(f"Cuda Memory Usage: {cuda_mem / (1024 ** 2):.2f} MB")

final_accuracy = sum(accuracy_list) / len(accuracy_list)
print(f"Final Accuracy: {final_accuracy:.2f}%")
