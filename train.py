import numpy as np
import torch
import torch.optim as optim
from model import MLP, train
from argparse import ArgumentParser
import json
from sklearn.model_selection import train_test_split

seed = 42
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudd.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specifyng the model

parser = ArgumentParser(description="Model config")
parser.add_argument("--input_dim", default=6, type=int)
parser.add_argument("--output_dim", default=1, type=int)
parser.add_argument("--hidden_dim", default=400, type=int)
parser.add_argument("--depth", default=4, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--base_lr", default=1e-5, type=float)
parser.add_argument("--max_lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=30000, type=int)
parser.add_argument("--eval_freq", default=25, type=int)

args = parser.parse_args()
config = vars(args)

# saving last config used
with open("last_config", "w") as fp: 
  json.dump(config, fp)

# Initializing the model
model = MLP(input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],).to(device)
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=config["base_lr"], max_lr=config["max_lr"], step_size_up=25, step_size_down=25
)
loss_fn = torch.nn.MSELoss()


# Data Loading
# TODO
X = np.load(X_path)
Y = np.load(Y_path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

dtype = torch.float
dataset_train = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train).type(dtype), torch.from_numpy(Y_train).type(dtype)
)
dataset_valid = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test).type(dtype), torch.from_numpy(Y_test).type(dtype)
)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)


if __name__ == "main":
  print(f"Training device = {device}")
  training_loss, validation_loss = train(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, device, epochs=config["epochs"], eval_feq=config["eval_freq"])
  np.save("training_loss", training_loss)
  np.save("validation_loss", validation_loss)
