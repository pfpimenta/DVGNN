# -*- coding: utf-8 -*-
""" Created on Mon May  5 15:14:25 2023
@author: Gorgen @Fuction：
（1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；
# """
import argparse
import configparser
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dvgcn import DVGCN
from lib.data_preparation import read_and_generate_dataset
from lib.utils import compute_val_loss, evaluate, predict, scaled_Laplacian
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import save_model_checkpoint, save_training_report

# set torch seed to assure reprocibility
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument(
    "--max_epoch", type=int, default=40, help="Epoch to run [default: 40]"
)
parser.add_argument(
    "--batch_size",
    type=int,
    # default=16,
    default=16,
    help="Batch Size during training [default: Tdrive: 8, los_loop: 16, PEMS08: 16]",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0005,
    help="Initial learning rate [default: 0.0005]",
)
parser.add_argument(
    "--momentum", type=float, default=0.9, help="Initial learning rate [default: 0.9]"
)
# parser.add_argument(
#     "--optimizer", default="adam", help="adam or momentum [default: adam]"
# )
# parser.add_argument("--length", type=int, default=60, help="Size of temporal : 12")
parser.add_argument(
    "--force", type=str, default=False, help="remove params dir", required=False
)
parser.add_argument(
    "--decay", type=float, default=0.92, help="decay rate of learning rate "
)
parser.add_argument(
    "--data_name",
    type=str,
    default="Tdrive",
    help="The dataset name: Tdrive, Los_loop, PEMS08 ",
    required=False,
)
parser.add_argument(
    "--num_point",
    type=int,
    default=64,
    help="road or grid Point Number 64,207,170 ",
    required=False,
)
parser.add_argument("--use_mixhop", action="store_true")
parser.add_argument("--gcn", dest="use_mixhop", action="store_false")
parser.set_defaults(use_mixhop=False)
parser.add_argument("--dynamic_graph", action="store_true")
parser.add_argument("--static_graph", dest="dynamic_graph", action="store_false")
parser.set_defaults(dynamic_graph=True)

# parse arguments
FLAGS = parser.parse_args()
decay = FLAGS.decay
dataset_name = FLAGS.data_name
# Length = FLAGS.length # not used
batch_size = FLAGS.batch_size
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate
# optimizer = FLAGS.optimizer # not used
# num_of_vertices = FLAGS.num_point # not used
use_mixhop = FLAGS.use_mixhop

# TODO use to select if Adj matrix used will be the dynamic or the static one
dynamic_graph = FLAGS.dynamic_graph


# if dataset_name == "Tdrive":
#     points_per_hour = 12
#     num_of_features = 1
#     merge = False
#     num_for_predict = 12
#     num_of_weeks = 2
#     num_of_days = 1
#     num_of_hours = 2
# if dataset_name == "Los_loop":
#     points_per_hour = 6
#     num_of_features = 1
#     merge = False
#     num_for_predict = 12
#     num_of_weeks = 2
#     num_of_days = 1
#     num_of_hours = 2
# if dataset_name == "PEMS08":
#     points_per_hour = 12
#     num_of_features = 3
#     merge = False
#     num_for_predict = 12
#     num_of_weeks = 2
#     num_of_days = 1
#     num_of_hours = 2


model_name = "DVGCN_%s" % dataset_name

wdecay = 0.000

# set device (CUDA or cpu)
print("Torch version: " + torch.__version__)
print("Is CUDA available? " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = torch.device(FLAGS.device)
else:
    device = torch.device("cpu")
# device = torch.device("cpu")  # DEBUG
print(f"Using device {device}")
# breakpoint() # TODO use GPU


print("Model is %s" % (model_name))

prediction_path = "DVGCN_prediction_%s" % dataset_name


# def get_data_loaders(dataset_name: str, batch_size: int, dataset_params: Dict[str, Any]) -> Tuple(DataLoader, DataLoader, DataLoader, np.ndarray, torch.Tensor):
# TODO move to another file
def get_data_loaders(
    dataset_name: str,
    batch_size: int,
    dataset_params: Dict[str, Any],
    device: torch.device,
):
    # get data paths
    data_folderpath = Path(__file__).parent.resolve() / "data/"
    adj_matrix_csv_filepath = data_folderpath / (
        "%s/%s_adj.csv" % (dataset_name, dataset_name)
    )
    graph_signal_matrix_filename = data_folderpath / (
        "%s/%s.npz" % (dataset_name, dataset_name)
    )
    generated_adj_filepath = (
        Path(__file__).parent.resolve()
        / f"generated_adj/dynamic_{dataset_name}_adj.npy"
    )
    # read all data from graph signal matrix file
    print("Reading data...")
    # load adj matrix
    adj = np.mat(pd.read_csv(adj_matrix_csv_filepath, header=None), dtype=float)
    adjs = scaled_Laplacian(adj)
    supports = (torch.tensor(adjs)).type(torch.float32).to(device)
    # Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data = read_and_generate_dataset(
        dataset_name,
        graph_signal_matrix_filename,
        generated_adj_filepath,
        dataset_params["num_of_weeks"],
        dataset_params["num_of_days"],
        dataset_params["num_of_hours"],
        dataset_params["num_for_predict"],
        dataset_params["points_per_hour"],
        dataset_params["merge"],
    )
    # test set ground truth
    true_value = all_data["test"]["target"]
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data["train"]["week"]),
            torch.Tensor(all_data["train"]["day"]),
            torch.Tensor(all_data["train"]["recent"]),  # hour
            torch.Tensor(all_data["train"]["target"]),
            torch.Tensor(all_data["train"]["recent_adj"]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data["val"]["week"]),
            torch.Tensor(all_data["val"]["day"]),
            torch.Tensor(all_data["val"]["recent"]),  # hour
            torch.Tensor(all_data["val"]["target"]),
            torch.Tensor(all_data["val"]["recent_adj"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data["test"]["week"]),
            torch.Tensor(all_data["test"]["day"]),
            torch.Tensor(all_data["test"]["recent"]),  # hour
            torch.Tensor(all_data["test"]["target"]),
            torch.Tensor(all_data["test"]["recent_adj"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, true_value, supports


# TODO move to another file
def evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    true_value: np.ndarray,
    supports: torch.Tensor,
    device: torch.device,
    epoch: int,
):
    # Evaluate on the test data
    start_time_test = time()
    test_rmse, test_mae, test_mape = evaluate(
        model, test_loader, true_value, supports, device, epoch
    )
    end_time_test = time()
    test_time = np.mean(end_time_test - start_time_test)
    test_stats = {
        "test_time": test_time,
        "rmse": test_rmse,
        "mae": test_mae,
        "mape": test_mape,
    }
    # print("Test time: %.2f" % test_time)
    return test_stats


# TODO move to another file
def train():
    clip = 5
    val_loss_list = []
    train_loss_list = []
    train_time = []
    rmse = []
    mae = []
    mape = []
    training_start_timestamp = str(datetime.now().strftime("%Y_%m_%d_%Hh%M"))
    print(f"training_start_timestamp: {training_start_timestamp}\n\n")

    save_model_checkpoint(
        model=model, epoch=0, training_start_timestamp=training_start_timestamp
    )
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        # for train_w, train_d, train_r, train_t, train_adj_r in train_loader:
        for i, batch in enumerate(tqdm(train_loader, desc="Training", file=sys.stdout)):
            train_w, train_d, train_r, train_t, train_adj_r = batch
            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            train_adj_r = train_adj_r.to(device)

            model.train()  # train pattern
            optimizer.zero_grad()  # grad to 0

            output, _, A = model(train_w, train_d, train_r, supports, train_adj_r)

            loss = loss_function(output, train_t)
            # backward p
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        train_loss_list.append(train_l)
        print(
            "epoch step: %s, training loss: %.2f, time: %.2fs"
            % (epoch, train_l, end_time_train - start_time_train)
        )
        train_time.append(end_time_train - start_time_train)

        # compute validation loss
        valid_loss = compute_val_loss(
            model, val_loader, loss_function, supports, device, epoch
        )
        val_loss_list.append(valid_loss)

        # evaluate the model on testing set
        rmse1, mae1, mape1 = evaluate(
            model, test_loader, true_value, supports, device, epoch
        )

        rmse1 = round(rmse1, 4)
        mae1 = round(mae1, 4)
        mape1 = round(mape1, 4)

        rmse.append(rmse1)
        mae.append(mae1)
        mape.append(mape1)

        # save model checkpoint
        save_model_checkpoint(
            model=model, epoch=epoch, training_start_timestamp=training_start_timestamp
        )

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))
    print("The min rmse is : " + str(min(rmse)))
    print("The min rmse epoch is : " + str(rmse.index(min(rmse))))
    print("The min mae is : " + str(min(mae)))
    print("The min mae epoch is : " + str(mae.index(min(mae))))
    print("The min mape is : " + str(min(mape)))
    print("The min mape epoch is : " + str(mape.index(min(mape))))

    training_stats = {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_time": train_time,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
    return model, training_start_timestamp, training_stats


if __name__ == "__main__":

    # get dataset params
    data_folderpath = Path(__file__).parent.parent.parent.resolve() / "data/"
    dataset_params_filepath = (
        data_folderpath / "Transportation" / "dataset_params" / f"{dataset_name}.conf"
    )
    print("Reading dataset params configuration file: %s" % (dataset_params_filepath))
    dataset_params_configparser = configparser.ConfigParser()
    dataset_params_configparser.read(dataset_params_filepath)
    dataset_params = {
        "num_for_predict": int(
            dataset_params_configparser["params"]["num_for_predict"]
        ),
        "num_of_weeks": int(dataset_params_configparser["params"]["num_of_weeks"]),
        "num_of_days": int(dataset_params_configparser["params"]["num_of_days"]),
        "num_of_hours": int(dataset_params_configparser["params"]["num_of_hours"]),
        "num_of_features": int(
            dataset_params_configparser["params"]["num_of_features"]
        ),
        "points_per_hour": int(
            dataset_params_configparser["params"]["points_per_hour"]
        ),
        "merge": dataset_params_configparser["params"].getboolean("merge"),
    }
    # load dataset
    train_loader, val_loader, test_loader, true_value, supports = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        dataset_params=dataset_params,
        device=device,
    )

    # initialize model
    model_params = {
        "c_in": dataset_params["num_of_features"],
        "c_out": 64,
        "num_nodes": num_nodes,
        "week": 24,
        "day": 12,
        "recent": 24,
        "K": 3,
        "Kt": 3,
        "use_mixhop": use_mixhop,
        "adjacency_powers": [0, 1, 2, 3],
        "device": str(device),
    }
    print(f"Using model params: {model_params}")
    model = DVGCN(**model_params)
    model.to(device)  # to cuda

    # loss function MSE
    loss_function = nn.MSELoss()

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)

    # calculate origin loss in epoch 0
    compute_val_loss(model, val_loader, loss_function, supports, device, epoch=0)

    # compute testing set MAE, RMSE, MAPE before training
    evaluate(model, test_loader, true_value, supports, device, epoch=0)

    # train model
    # TODO modularizar essa parte aqui
    clip = 5
    val_loss_list = []
    train_loss_list = []
    train_time = []
    rmse = []
    mae = []
    mape = []
    training_start_timestamp = str(datetime.now().strftime("%Y_%m_%d_%Hh%M"))
    print(f"training_start_timestamp: {training_start_timestamp}\n\n")

    save_model_checkpoint(
        model=model, epoch=0, training_start_timestamp=training_start_timestamp
    )
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        # for train_w, train_d, train_r, train_t, train_adj_r in train_loader:
        for i, batch in enumerate(tqdm(train_loader, desc="Training", file=sys.stdout)):
            train_w, train_d, train_r, train_t, train_adj_r = batch
            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            train_adj_r = train_adj_r.to(device)

            model.train()  # train pattern
            optimizer.zero_grad()  # grad to 0

            output, _, A = model(train_w, train_d, train_r, supports, train_adj_r)

            loss = loss_function(output, train_t)
            # backward p
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        train_loss_list.append(train_l)
        print(
            "epoch step: %s, training loss: %.2f, time: %.2fs"
            % (epoch, train_l, end_time_train - start_time_train)
        )
        train_time.append(end_time_train - start_time_train)

        # compute validation loss
        valid_loss = compute_val_loss(
            model, val_loader, loss_function, supports, device, epoch
        )
        val_loss_list.append(valid_loss)

        # evaluate the model on testing set
        rmse1, mae1, mape1 = evaluate(
            model, test_loader, true_value, supports, device, epoch
        )

        rmse1 = round(rmse1, 4)
        mae1 = round(mae1, 4)
        mape1 = round(mape1, 4)

        rmse.append(rmse1)
        mae.append(mae1)
        mape.append(mape1)

        # save model checkpoint
        save_model_checkpoint(
            model=model, epoch=epoch, training_start_timestamp=training_start_timestamp
        )

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))
    print("The min rmse is : " + str(min(rmse)))
    print("The min rmse epoch is : " + str(rmse.index(min(rmse))))
    print("The min mae is : " + str(min(mae)))
    print("The min mae epoch is : " + str(mae.index(min(mae))))
    print("The min mape is : " + str(min(mape)))
    print("The min mape epoch is : " + str(mape.index(min(mape))))
    training_stats = {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_time": train_time,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }

    # Evaluate on the test data
    test_stats = evaluation(
        model=model,
        test_loader=test_loader,
        true_value=true_value,
        supports=supports,
        device=device,
        epoch=epoch,
    )

    # save training report
    training_params = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "decay": decay,
        "optimizer": str(optimizer),
        "device": str(device),
    }
    save_training_report(
        training_start_timestamp=training_start_timestamp,
        dataset_params=dataset_params,
        model_params=model_params,
        training_params=training_params,
        training_stats=training_stats,
        test_stats=test_stats,
    )

    # plot RMSE and MAE evolution
    # 绘制折线图
    fig = plt.figure()
    fig = plt.figure(figsize=(15, 8))  # 画柱形图
    ax1 = fig.add_subplot(111)
    yerr = np.linspace(0.05, 0.2, 10)
    x = np.linspace(1, epochs, epochs)
    plt.errorbar(
        x,
        training_stats["rmse"],
        marker="H",
        markersize=12,
        yerr=yerr[0],
        uplims=True,
        lolims=True,
        label="rmse",
    )
    plt.errorbar(
        x,
        training_stats["mae"],
        marker="D",
        markersize=10,
        yerr=yerr[1],
        uplims=True,
        lolims=True,
        label="mae",
    )

    plt.show()

    # save predictions on the test dataset:
    prediction, spatial_at, parameter_adj = predict(
        model, test_loader, supports, device
    )
    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        spatial_at=spatial_at,
        parameter_adj=parameter_adj,
        ground_truth=true_value,
    )
