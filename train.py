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
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--base_lr", default=1e-5, type=float)
parser.add_argument("--max_lr", default=5e-5, type=float)
parser.add_argument("--epochs", default=800, type=int)
parser.add_argument("--eval_freq", default=1, type=int)

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
    step_size_up=25,
    step_size_down=25,
)
loss_fn = torch.nn.MSELoss()


dtype = torch.float32
## Loading train data

with open("data/X_train.npy", "rb") as f:
    X_train = np.load(f)
with open("data/Y_train_averaged_10kpaths.npy", "rb") as f:
    Y_train_averaged = np.load(f)

# Initializing torch dataset and dataloader
dataset_train = {
    f"{i+1}k_paths": torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).type(dtype),
        torch.from_numpy(Y_train_averaged[:, i]).type(dtype),
    )
    for i in range(Y_train_averaged.shape[1])
}
train_loader = {
    f"{i+1}k_paths": torch.utils.data.DataLoader(
        dataset_train[f"{i+1}k_paths"],
        batch_size=config["batch_size_train"],
        shuffle=True,
    )
    for i in range(Y_train_averaged.shape[1])
}

## Loading validation data


with open("data/X_valid.npy", "rb") as f:
    X_valid = np.load(f)
with open("data/Y_valid.npy", "rb") as f:
    Y_valid = np.load(f)

# Initializing torch dataset and dataloader
dataset_valid = torch.utils.data.TensorDataset(
    torch.from_numpy(X_valid).type(dtype), torch.from_numpy(Y_valid).type(dtype)
)
valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=config["batch_size_valid"],
    shuffle=False,
)


if __name__ == "__main__":
    print(f"TRAINING ON DEVICE = {device}")
    print(config)
    # for nb_paths, loader in train_loader.items():
    #     model = MLP(
    #         input_dim=config["input_dim"],
    #         output_dim=config["output_dim"],
    #         hidden_dim=config["hidden_dim"],
    #         depth=config["depth"],
    #         normalization=config["normalization"],
    #     ).to(device)
    #     optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    #     scheduler = optim.lr_scheduler.CyclicLR(
    #         optimizer,
    #         base_lr=config["base_lr"],
    #         max_lr=config["max_lr"],
    #         step_size_up=50,
    #         step_size_down=50,
    #     )
    #     loss_fn = torch.nn.MSELoss()
    #     training_loss, validation_loss = train(
    #         model,
    #         optimizer,
    #         scheduler,
    #         loss_fn,
    #         loader,
    #         valid_loader,
    #         device,
    #         epochs=config["epochs"],
    #         eval_freq=config["eval_freq"],
    #         checkpoint=True,
    #         checkpoint_path=f"checkpoints/checkpoint_{nb_paths}_new.pth",
    #         training_loss_path=f"logs/training_loss_{nb_paths}_new",
    #         validation_loss_path=f"logs/validation_loss_{nb_paths}_new",
    #     )

    model = MLP(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        normalization=config["normalization"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        gamma=0.95,
        step_size=20,
    )
    training_loss, validation_loss = train(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_loader["1k_paths"],
        valid_loader,
        device,
        epochs=config["epochs"],
        eval_freq=config["eval_freq"],
        checkpoint=True,
        checkpoint_path=f"checkpoints/checkpoint_1k_paths_steplr.pth",
        training_loss_path=f"logs/training_loss_1k_paths_steplr",
        validation_loss_path=f"logs/validation_loss_1k_paths_steplr",
    )
