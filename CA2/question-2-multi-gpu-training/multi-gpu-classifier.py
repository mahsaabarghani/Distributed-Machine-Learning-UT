import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import time
import datetime
import os
import torchvision.datasets as dsets
from torch.nn.parallel import DistributedDataParallel as DDP

data_path = "/storage/dmls/stl10_data"
learning_rate = 0.001
training_epochs = 10
batch_size = 32

def setup(rank, world_size, master_port, backend, timeout):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)

def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


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

def load_data(rank, world_size):
    train_set = dsets.STL10(data_path, split="train", transform=transforms.Compose([transforms.ToTensor()]), download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, sampler=train_sampler, batch_size=batch_size,
        persistent_workers=True, num_workers=1, pin_memory=True
    )


    test_set = dsets.STL10(data_path, split="test", transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=world_size, rank=rank
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, sampler=test_sampler, batch_size=batch_size,
        persistent_workers=True, num_workers=1, pin_memory=True
    )
   
    return train_loader, test_loader

def train(rank, world_size, master_port, backend, timeout):
    setup(rank, world_size, master_port, backend, timeout)
    torch.cuda.set_device(rank)
    train_loader, test_loader = load_data(rank, world_size)
    model = STL10CNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    predictions_list = []
    labels_list = []

    start_time = time.time()
    for epoch in range(training_epochs):
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            count += 1
            if count % 50 == 0:
                total = 0
                correct = 0
                for images, labels in test_loader:
                    images, labels = images.to(rank), labels.to(rank)
                    labels_list.append(labels)
                    outputs = ddp_model(images)
                    predictions = torch.max(outputs, 1)[1].to(rank)
                    labels_list.append(labels)
                    correct += (predictions == labels).sum().item()
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += labels.size(0)

                accuracy = correct * 100 / total
                loss_list.append(loss.item())
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            if count % 500 == 0:
                print(f"Iteration: {count}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    end_time = time.time()
    cuda_mem = torch.cuda.max_memory_allocated(rank)
    print(f"Rank: {rank}, Training time: {end_time - start_time:.2f}s")
    print(f"Rank: {rank}, Max Memory Allocated: {cuda_mem / (1024 ** 2):.2f} MB")

    final_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"Rank: {rank}, Final Accuracy: {final_accuracy:.2f}%")



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    backends = ['gloo', 'nccl']
    timeout = datetime.timedelta(seconds=10)

    for backend in backends:
        print(f"Starting training with {backend}")
        start_time = time.time()
        mp.spawn(train, nprocs=world_size, args=(world_size, master_port, backend, timeout), join=True, start_method="spawn")
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f}s")