"""cs54-flwr: A Flower / PyTorch app."""

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter

# Efficient transforms for medical imaging
PYTORCH_TRANSFORMS = Compose([
    ToTensor(),
    ColorJitter(brightness=0.1, contrast=0.1),  # Light augmentation
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variables
NUM_CLASSES = 5
LABEL_MAP = None

class EfficientBlock(nn.Module):
    """Lightweight residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Efficient single conv design
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed to non-inplace
        
        # Simple skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))  # Using class ReLU
        out = out + self.shortcut(x)  # Changed from += to +
        return self.relu(out)  # Using class ReLU

class Net(nn.Module):
    """Efficient model for cervical cancer classification"""
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)  # Changed to non-inplace
        
        # Efficient feature extraction
        self.layer1 = self._make_layer(32, 64, 2, stride=2)    # 56x56
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 28x28
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 14x14
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, NUM_CLASSES)
        
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(EfficientBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(EfficientBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Using class ReLU
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    global LABEL_MAP
    # Convert images to tensors and apply transforms
    batch["image"] = torch.stack([PYTORCH_TRANSFORMS(img) for img in batch["image"]])
    
    # Handle string labels - extract first element if it's a list/array
    labels = [label[0] if isinstance(label, (list, tuple, np.ndarray)) else label 
              for label in batch["label"]]
    
    # Create a mapping for string labels to integers if not already created
    if LABEL_MAP is None:
        unique_labels = sorted(set(labels))
        LABEL_MAP = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"\nLabel mapping: {LABEL_MAP}")
        print(f"Number of classes: {NUM_CLASSES}\n")
    
    # Convert string labels to numeric indices
    numeric_labels = [LABEL_MAP[label] for label in labels]
    batch["label"] = torch.tensor(numeric_labels, dtype=torch.long)
    return batch

def load_data(partition_id: int, num_partitions: int):
    """Load partition of Cervical Cancer dataset."""
    # Initialize FederatedDataset for this worker
    partitioner = DirichletPartitioner(num_partitions=num_partitions,alpha=10,partition_by="label")
    fds = FederatedDataset(
        dataset="Alwaly/Cervical_Cancer-cancer",
        partitioners={"train": partitioner},
    )
    
    # Load this worker's partition
    partition = fds.load_partition(partition_id)
    
    # Split into train and test (80/20)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Apply transforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # Create data loaders
    trainloader = DataLoader(
        partition_train_test["train"], 
        batch_size=32, 
        shuffle=True,
        drop_last=True  # Prevent issues with last batch being smaller
    )
    testloader = DataLoader(
        partition_train_test["test"], 
        batch_size=32,
        drop_last=True
    )
    
    return trainloader, testloader

def train(net, trainloader, epochs, device):
    """Efficient training loop"""
    net.to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / len(trainloader)

def test(net, testloader, device):
    """Efficient evaluation"""
    net.to(device)
    net.eval()
    criterion = nn.NLLLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Metrics aggregation functions
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate metrics weighted by number of samples."""
    if not metrics:
        return {}
    
    # Unpack the metrics
    weights, values = zip(*[(weight, value) for weight, value in metrics])
    weights = np.array(weights)
    
    # Calculate weighted average for each metric
    weighted_metrics = {}
    for metric in values[0].keys():
        metric_values = np.array([val[metric] for val in values])
        weighted_metrics[metric] = float(np.average(metric_values, weights=weights))
    
    return weighted_metrics
