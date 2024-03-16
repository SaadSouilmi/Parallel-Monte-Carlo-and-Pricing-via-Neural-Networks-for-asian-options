import numpy as np
import torch
import torch.optim as optim
from model import MLP, train
from sklearn.model_selection import train_test_split


seed = 42
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudd.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specifyng the model
config = dict(input_dim=5,
              output_dim=1,
              hidden_dim=400,
              depth=4,
              batch_size=512,
              lr=1e-3,
              base_lr=1e-5,
              max_lr=1e-3,
              epochs=30000,
              eval_freq=25)
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
