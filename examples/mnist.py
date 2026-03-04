#!/usr/bin/env python3
"""MNIST training example with gsql experiment tracking.

3 lines to set up tracking, 1 line per epoch to log metrics.
After training: `gsql ~/.gsql/track.db` to browse results.

Requirements: pip install torch torchvision
"""

import sys
import os

# Add parent dir so we can import gsql_track
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from gsql_track import GsqlTrack

# --- Model ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- Training ---

def main():
    lr = 0.001
    batch_size = 64
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Set up tracking
    t = GsqlTrack("mnist")
    run = t.start_run("cnn-001")
    run.log_params({"lr": lr, "batch_size": batch_size, "epochs": epochs, "device": device})

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                correct += output.argmax(1).eq(target).sum().item()
                total += data.size(0)
        test_loss /= total
        test_acc = correct / total

        # 2) Log metrics (one line!)
        run.log(step=epoch, train_loss=train_loss, train_acc=train_acc,
                test_loss=test_loss, test_acc=test_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # 3) Finish
    run.finish()
    print(f"\nRun {run.id} complete. Browse with: gsql ~/.gsql/track.db")


if __name__ == "__main__":
    main()
