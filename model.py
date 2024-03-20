import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
from collections import deque


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        hidden_dim,
        activation=nn.SiLU(),
        normalization=None,
    ):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        if normalization == "batch":
            self.normalization = nn.BatchNorm1d(hidden_dim)
        elif normalization == "layer":
            self.normalization = nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.normalization is not None:
                x = self.normalization(x)
        return nn.functional.relu(
            self.output_layer(x)
        )  # We run the last layer through a relu since price is positive


# Training loop
def train(
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    valid_loader,
    device,
    epochs=100,
    eval_freq=5,
    checkpoint=True,
):
    training_loss = deque()
    validation_loss = deque()
    best_validation_loss = float("inf")
    desc = "Training Loop"
    with tqdm.tqdm(total=epochs, desc=desc, position=0, leave=True) as progress_bar:
        for epoch in range(epochs):
            # Gradient descent over train dataloader
            train_loss = 0
            valid_loss = 0
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs.flatten(), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss
            train_loss = train_loss / (i + 1)
            training_loss.append(train_loss.item())

            if epoch % eval_freq == 0:
                valid_loss = 0
                model.eval()
                with torch.no_grad():
                    for inputs, targets in valid_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs.flatten(), targets)
                        valid_loss += loss
                valid_loss = valid_loss / len(valid_loader)
                validation_loss.append(valid_loss.item())

                # Setting checkpoints
                if checkpoint and valid_loss < best_validation_loss:
                    print("Checkpoint")
                    best_validation_loss = valid_loss
                    torch.save(model.state_dict(), "checkpoint.pth")
                # Caching results
                np.save("training_loss", training_loss)
                np.save("validation_loss", validation_loss)

            desc = f"Epoch: {epoch}, lr = {scheduler.get_last_lr()} , training_loss = {train_loss}, validation_loss = {valid_loss}"
            scheduler.step()
            print(desc)
            # progress_bar.set_description(desc)
            # progress_bar.update(1)

    return training_loss, validation_loss
