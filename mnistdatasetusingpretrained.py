import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=32):
        super(Autoencoder, self).__init__()


        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def encode(self, x):
        return self.encoder(x)


    def decode(self, x):
        return self.decoder(x)




def train_autoencoder(model, train_loader, epochs=10, lr=0.001, device='cpu'):
    """Train the autoencoder"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    model.to(device)
    losses = []


    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            # Flatten images and move to device
            data = data.view(data.size(0), -1).to(device)


            # Forward pass
            output = model(data)
            loss = criterion(output, data)


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


    return losses




def visualize_reconstruction(model, test_loader, device='cpu', n_images=10):
    """Visualize original vs reconstructed images"""
    model.eval()


    # Get a batch of test data
    data, _ = next(iter(test_loader))
    data = data[:n_images]


    # Reconstruct
    with torch.no_grad():
        data_flat = data.view(data.size(0), -1).to(device)
        reconstructed = model(data_flat)
        reconstructed = reconstructed.cpu().view(-1, 28, 28)


    # Plot
    fig, axes = plt.subplots(2, n_images, figsize=(15, 3))
    for i in range(n_images):
        # Original
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)


        # Reconstructed
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)


    plt.tight_layout()
    plt.savefig('autoencoder_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()




def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    # Hyperparameters
    batch_size = 128
    epochs = 10
    learning_rate = 0.001
    encoding_dim = 32


    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )


    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Initialize model
    model = Autoencoder(input_dim=784, encoding_dim=encoding_dim)
    print(f'\nModel Architecture:\n{model}\n')


    # Train
    print('Training autoencoder...')
    losses = train_autoencoder(model, train_loader, epochs=epochs, lr=learning_rate, device=device)


    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()


    # Visualize reconstructions
    print('\nGenerating reconstructions...')
    visualize_reconstruction(model, test_loader, device=device, n_images=10)


    # Save model
    torch.save(model.state_dict(), 'autoencoder.pth')
    print('\nModel saved to autoencoder.pth')




if __name__ == '__main__':
    main()
