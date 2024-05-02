import numpy as np 
import torch  
import torch.nn as nn 
import torch.optim as optim  
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define data augmentation and normalization for both training and testing data
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate images by 10 degrees
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images with mean 0.5 and standard deviation 0.5
])

test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

# Download and load the training data
train_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=train_transform)
test_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=test_transform)

# Define the data loaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Determine the shape of the input images
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_shape = images[0].shape
print("Image Shape:", img_shape)

# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 input channel, 32 output channels, 3x3 kernel size, stride of 1
        self.conv2 = nn.Conv2d(32, 48, 3, 1)  # 32 input channels, 48 output channels, 3x3 kernel size, stride of 1
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel size and stride of 2
        self.dropout = nn.Dropout(0.5)  # Dropout layer with dropout probability of 0.5
        self.fc1 = nn.Linear(5*5*48, 500)  # Fully connected layer with 5x5x48 input features and 500 output features
        self.fc2 = nn.Linear(500, 10)  # Fully connected layer with 500 input features and 10 output features

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # First convolutional layer followed by ReLU activation and max pooling
        x = self.pool(nn.functional.relu(self.conv2(x)))  # Second convolutional layer followed by ReLU activation and max pooling
        x = self.dropout(x)  # Apply dropout regularization
        x = torch.flatten(x, 1)  # Flatten the input tensor
        x = nn.functional.relu(self.fc1(x))  # Fully connected layer followed by ReLU activation
        x = nn.functional.softmax(self.fc2(x), dim=1)  # Output layer with softmax activation
        return x

# Initialize the model
model = CNN()

# Print the model summary
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate of 0.001

# Training the model
epochs = 5 
train_losses, test_losses = [], [] 

for epoch in range(epochs):  # Loop over the epochs
    running_loss = 0  # Initialize running loss for the current epoch
    model.train()  # Set the model to training mode
    for images, labels in train_loader:  # Iterate over the training data
        optimizer.zero_grad()  # Zero the gradients
        log_ps = model(images)  # Forward pass
        loss = criterion(log_ps, labels)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Accumulate the running loss
        
    else:  # After finishing one epoch of training
        test_loss = 0  # Initialize test loss
        accuracy = 0  # Initialize accuracy
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for images, labels in test_loader:  # Iterate over the test data
                log_ps = model(images)  # Forward pass
                test_loss += criterion(log_ps, labels)  # Accumulate test loss
                ps = torch.exp(log_ps)  # Compute predicted probabilities
                top_p, top_class = ps.topk(1, dim=1)  # Get predicted class
                equals = top_class == labels.view(*top_class.shape)  # Check if predicted labels match true labels
                accuracy += torch.mean(equals.type(torch.FloatTensor))  # Calculate accuracy
        
        train_losses.append(running_loss/len(train_loader))  # Calculate average training loss
        test_losses.append(test_loss/len(test_loader))  # Calculate average test loss

        print(f"Epoch: {epoch+1}/{epochs}.. ", 
              f"Training Loss: {running_loss/len(train_loader):.3f}.. ", 
              f"Test Loss: {test_loss/len(test_loader):.3f}.. ",  
              f"Test Accuracy: {accuracy/len(test_loader):.3f}") 

# Plotting the training and validation loss
plt.figure(figsize=(12, 5))  
plt.plot(train_losses, label='Training loss') 
plt.plot(test_losses, label='Validation loss')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.legend(frameon=False)  
plt.show()

# Testing the model
correct = 0
total = 0
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    for images, labels in test_loader:  # Iterate over test data
        log_ps = model(images)  # Forward pass
        _, predicted = torch.max(log_ps.data, 1)  # Get predicted labels
        total += labels.size(0)  # Accumulate the total number of images
        correct += (predicted == labels).sum().item()  # Count correct predictions

accuracy = 100 * correct / total  
print('Accuracy: %.2f %%' % accuracy) 

# Predict and show the first 20 images
images, labels = next(iter(test_loader))  # Get a batch of test images and labels
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    log_ps = model(images)  # Forward pass
    ps = torch.exp(log_ps)  # Compute predicted probabilities
    _, predicted = torch.max(ps, 1)  # Get predicted labels

plt.figure(figsize=(12, 5))  
for i in range(20):  
    plt.subplot(2, 10, i + 1) 
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')  
    plt.axis('off')  
    plt.title(f'Pred: {predicted[i].item()}, Actual: {labels[i].item()}')  
plt.show()