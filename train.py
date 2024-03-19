import numpy as np
import torch
import torch.optim as optim
from model import MLP, train
from argparse import ArgumentParser
from dataset import PayoffDataset
import json

seed = 42
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specifyng the model

parser = ArgumentParser(description="Model config")
parser.add_argument("--input_dim", default=6, type=int)
parser.add_argument("--output_dim", default=1, type=int)
parser.add_argument("--hidden_dim", default=400, type=int)
parser.add_argument("--depth", default=4, type=int)
parser.add_argument("--normalization", default="layer", type=str)
parser.add_argument("--batch_size_train", default=2048, type=int)
parser.add_argument("--batch_size_valid", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--base_lr", default=1e-5, type=float)
parser.add_argument("--max_lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--eval_freq", default=100, type=int)

args = parser.parse_args()
config = vars(args)

# saving last config used
with open("last_config", "w") as fp:
    json.dump(config, fp)

# Initializing the model
model = MLP(
    input_dim=config["input_dim"],
    output_dim=config["output_dim"],
    hidden_dim=config["hidden_dim"],
    depth=config["depth"],
    normalization=config["normalization"],
).to(device)
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=config["base_lr"],
    max_lr=config["max_lr"],
    step_size_up=5,
    step_size_down=5,
)
loss_fn = torch.nn.MSELoss()


# Data Loading
print("########## LOADING TRAIN DATA ##########")
dtype = torch.float32
# train data
with open("X_train.npy", "rb") as f:
    X_train = np.load(f)
with open("Y_train.npy", "rb") as f:
    Y_train = np.load(f)
dataset_train = PayoffDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=config["batch_size_train"],
    shuffle=True,  # num_workers=8
)
print("########## TRAIN DATA LOADED ##########")
print("    ")
# test data
print("########## LOADING VALIDATION DATA ##########")
with open("X_valid.npy", "rb") as f:
    X_valid = np.load("X_valid.npy")
with open("Y_valid.npy", "rb") as f:
    Y_valid = np.load("Y_valid.npy")
dataset_valid = torch.utils.data.TensorDataset(
    torch.from_numpy(X_valid).type(dtype), torch.from_numpy(Y_valid).type(dtype)
)
valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=config["batch_size_valid"],
    shuffle=False,  # num_workers=8
)
print("########## VALIDATION DATA LOADED ##########")
print("    ")

if __name__ == "__main__":
    print(f"TRAINING ON DEVICE = {device}")
    print(config)
    training_loss, validation_loss = train(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_loader,
        valid_loader,
        device,
        epochs=config["epochs"],
        eval_freq=config["eval_freq"],
    )
    np.save("training_loss", training_loss)
    np.save("validation_loss", validation_loss)
