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
import torch.optim
from dvgcn import DVGCN
from lib.utils import compute_val_loss, evaluate, predict
from utils import evaluation, get_data_loaders, save_training_report, train

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
parser.add_argument(
    "--mixhop_neighborhood",
    type=int,
    default=2,
    help="range of the MixHop neighborhood",
    required=False,
)
parser.add_argument(
    "--dynamic_graph",
    action="store_true",
    help="select if Adj matrix used will be the dynamic or the static one",
)
parser.add_argument("--static_graph", dest="dynamic_graph", action="store_false")
parser.set_defaults(dynamic_graph=True)

# parse arguments
FLAGS = parser.parse_args()
decay = FLAGS.decay
dataset_name = FLAGS.data_name
batch_size = FLAGS.batch_size
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate
use_mixhop = FLAGS.use_mixhop
mixhop_neighborhood = list(range(FLAGS.mixhop_neighborhood + 1))
use_dynamic_graph = FLAGS.dynamic_graph

# Length = FLAGS.length # not used
# optimizer = FLAGS.optimizer # not used
# num_of_vertices = FLAGS.num_point # not used


if __name__ == "__main__":
    model_name = "DVGCN_%s" % dataset_name

    wdecay = 0.000

    # set device (CUDA or cpu)
    print("Torch version: " + torch.__version__)
    print("Is CUDA available? " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        device = torch.device(FLAGS.device)
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")  # force to CPU
    print(f"Using device {device}")

    print("Model is %s" % (model_name))

    prediction_path = "DVGCN_prediction_%s" % dataset_name

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
        "adjacency_powers": mixhop_neighborhood,
        "device": str(device),
    }
    print(f"Using model params: {model_params}")
    model = DVGCN(**model_params)
    model.to(device)  # to cuda

    # loss function MSE
    loss_function = nn.MSELoss()

    # calculate origin loss in epoch 0
    compute_val_loss(
        net=model,
        val_loader=val_loader,
        loss_function=loss_function,
        supports=supports,
        device=device,
        epoch=0,
        use_dynamic_graph=use_dynamic_graph,
    )

    # compute testing set MAE, RMSE, MAPE before training
    evaluate(
        model,
        test_loader,
        true_value,
        supports,
        device,
        epoch=0,
        use_dynamic_graph=use_dynamic_graph,
    )

    # Optimizer: Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=wdecay
    )

    # train model
    model, training_start_timestamp, training_stats = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        supports=supports,
        true_value=true_value,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        decay=decay,
        epochs=epochs,
        use_dynamic_graph=use_dynamic_graph,
    )

    # Evaluate on the test data
    test_stats = evaluation(
        model=model,
        test_loader=test_loader,
        true_value=true_value,
        supports=supports,
        device=device,
        epoch=epochs,
        use_dynamic_graph=use_dynamic_graph,
    )

    # save training report
    training_params = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "decay": decay,
        "use_dynamic_graph": use_dynamic_graph,
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
        net=model,
        test_loader=test_loader,
        supports=supports,
        device=device,
        use_dynamic_graph=use_dynamic_graph,
    )
    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        spatial_at=spatial_at,
        parameter_adj=parameter_adj,
        ground_truth=true_value,
    )
