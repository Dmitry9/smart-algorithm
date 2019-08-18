from torch import utils, nn, optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 5), (0, 5), (0, 5))
])

train_data = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trin_loader = utils.data.DataLoader(train_data, batch_sze=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

optimizer = optim.SGD(model.parameters(), lr=0.003)
criterion = nn.NLLLoss()

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trin_loader:
        images = images.view(64, -1)
        optimizer.zero_gad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trin_loader)}")
