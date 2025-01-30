import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)

train_dataset = datasets.MNIST('data/mnist', download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((28, 28)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.1300], std=[0.0700])
                              ]))

test_dataset = datasets.MNIST('data/mnist', download=False,
                        transform=transforms.Compose([
                            transforms.Resize((28, 28)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.1300], std=[0.0700])
                        ]),
                        target_transform=lambda x: torch.tensor(x)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=10000,
    shuffle=False,
    num_workers=4
)

num_epochs = 10
learning_rate = 0.001
momentum = 0.9

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print('Model architecture:', model)
print('Number of parameters:', sum(p.numel() for p in model.parameters()))

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss: {loss.item():.4f}')
    
    print('-' * 10)

print('Training complete!')

model.eval()
correct = 0
total = 0

for batch_idx, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        total += len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

print('Test accuracy: %.2f%%' % (correct / total * 100))
