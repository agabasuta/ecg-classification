import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from models.models2d import MobileNetV2
from dataloaders.dataset2d import EcgDataset2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for batch in dataloader:
        inputs, labels = batch["image"].to(device), batch["class"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["image"].to(device), batch["class"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return avg_loss, acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = json.loads(open(args.config).read())

    # Model and optimizer
    model = MobileNetV2(num_classes=config["num_classes"]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Dataset
    dataset = EcgDataset2D(config["train_ann_path"], config["class_mapping_path"])
    dataloader = dataset.get_dataloader(batch_size=config["batch_size"])

    # Training loop
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_one_epoch(model, dataloader, optimizer, DEVICE)
