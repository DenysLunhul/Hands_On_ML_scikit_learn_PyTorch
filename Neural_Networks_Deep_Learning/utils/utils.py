import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_cifar10_loaders(data_root="./data", batch_size=32, val_size=5000, num_workers=4):
    raw = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transforms.ToTensor())
    all_images = torch.stack([img for img, _ in raw])
    mean = all_images.mean(dim=[0, 2, 3])
    std = all_images.std(dim=[0, 2, 3])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
    train_data, valid_data = random_split(train_data, [len(train_data) - val_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, mean, std


def he_initialization(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


def build_mlp(n_hidden, n_neurons, n_inputs, n_outputs, dropout_p):
    layers = [nn.Flatten(), nn.Linear(n_inputs, n_neurons), nn.BatchNorm1d(n_neurons), nn.ReLU(), nn.Dropout(dropout_p)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(n_neurons, n_neurons), nn.BatchNorm1d(n_neurons), nn.ReLU(), nn.Dropout(dropout_p)]
    layers += [nn.Linear(n_neurons, n_outputs)]
    for module in layers:
        he_initialization(module)
    return nn.Sequential(*layers)


def evaluate(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    model.train()
    return metric.compute().item()


def train(model, optimizer, criterion,
          train_loader, valid_loader, metric,
          n_epochs, n_iter_no_improvements, scheduler=None, metric_direction="maximize"):
    model.train()
    if metric_direction == "maximize":
        best_val_score = 0
    elif metric_direction == "minimize":
        best_val_score = float('inf')
    iter_no_improvements = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        mean_loss = epoch_loss / len(train_loader)
        val_score = evaluate(model, valid_loader, metric)
        print(f"Epoch: {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}, Val Score: {val_score:.4f}")

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        if metric_direction == "maximize":
            if val_score > best_val_score:
                best_val_score = val_score
                iter_no_improvements = 0
            else:
                iter_no_improvements += 1

        elif metric_direction == "minimize":
            if val_score < best_val_score:
                best_val_score = val_score
                iter_no_improvements = 0
            else:
                iter_no_improvements += 1

        else:
            raise ValueError("Not valid direction")

        if iter_no_improvements >= n_iter_no_improvements:
            print(f"Validation score has not improved for {n_iter_no_improvements} epochs, stopping training")
            return best_val_score

    return best_val_score
