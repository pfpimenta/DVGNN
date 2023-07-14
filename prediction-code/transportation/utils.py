# -*- coding: utf-8 -*-

import json
import sys
from datetime import datetime
from pathlib import Path

# from utils import save_model_checkpoint
from time import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.optim
from lib.data_preparation import read_and_generate_dataset
from lib.utils import compute_val_loss, evaluate, scaled_Laplacian
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_project_root_folderpath() -> str:
    project_folderpath = Path(__file__).parent.parent.parent.resolve()
    return project_folderpath


def get_training_results_dir(training_start_timestamp: str) -> str:
    project_folderpath = get_project_root_folderpath()
    training_results_dir = (
        project_folderpath / "training_results" / f"{training_start_timestamp}"
    )
    training_results_dir.mkdir(parents=True, exist_ok=True)
    return training_results_dir


def save_training_report(
    training_start_timestamp: str,
    dataset_params: Dict[str, Any],
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    training_stats: Dict[str, List],
    test_stats: Dict[str, float],
):
    """Saves training information in a JSON file"""
    training_report = {
        "training_start_timestamp": training_start_timestamp,
        "dataset_params": dataset_params,
        "model_params": model_params,
        "training_params": training_params,
        "training_stats": training_stats,
        "test_stats": test_stats,
    }
    training_results_dir = get_training_results_dir(
        training_start_timestamp=training_start_timestamp
    )
    json_filepath = (
        training_results_dir / f"training_report_{training_start_timestamp}.json"
    )
    with open(json_filepath, "w") as f:
        json.dump(training_report, f)
    print(f"Saved complete training report on {json_filepath}")
    # save training report txt
    text_report = f"Model trained on {training_start_timestamp}"
    # text_report += f"\n* model.pt filepath: {trained_model_filepath}"
    text_report += f"\n* Complete JSON report filepath: {json_filepath}"
    text_report += (
        f"\n* Average time per epoch: {np.mean(training_stats['train_time'])}"
    )
    text_report += f"\n* Minumum MAE: {np.min(training_stats['mae'])}"
    text_report += f"\n* Minimum RMSE: {np.min(training_stats['rmse'])}"
    text_report += f"\n* Minimum MAPE: {np.min(training_stats['mape'])}"
    text_report += "\n"
    txt_filepath = (
        training_results_dir / f"training_report_{training_start_timestamp}.txt"
    )
    with open(txt_filepath, "w") as f:
        f.write(text_report)
    print(f"Saved summarized training report on {txt_filepath}")


def save_model_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    training_start_timestamp: str,
    # evaluation_overview: str,
    # performance_dict: Dict[str, Any],
    # training_loss_list: np.ndarray,
) -> None:
    # saves model at training_results/{timestamp}/{epoch}/model.pt
    training_results_dir = (
        get_training_results_dir(training_start_timestamp=training_start_timestamp)
        / f"epoch_{epoch}"
    )
    training_results_dir.mkdir(parents=True, exist_ok=True)
    trained_model_filepath = training_results_dir / "model.pt"
    torch.save(model.state_dict(), trained_model_filepath)
    print(f"Saved {trained_model_filepath}")


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
    # load static adjacency matrix
    static_adj = np.mat(pd.read_csv(adj_matrix_csv_filepath, header=None), dtype=float)
    adjs = scaled_Laplacian(static_adj)
    supports = (torch.tensor(adjs)).type(torch.float32).to(device)
    static_adj = (torch.tensor(static_adj)).type(torch.float32).to(device)
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
    return train_loader, val_loader, test_loader, true_value, supports, static_adj


# TODO move to another file
def evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    true_value: np.ndarray,
    static_adj: torch.Tensor,
    device: torch.device,
    epoch: int,
    use_dynamic_graph: bool = True,
):
    # Evaluate on the test data
    start_time_test = time()
    test_rmse, test_mae, test_mape = evaluate(
        net=model,
        test_loader=test_loader,
        true_value=true_value,
        static_adj=static_adj,
        device=device,
        epoch=epoch,
        use_dynamic_graph=use_dynamic_graph,
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
def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    static_adj: torch.Tensor,
    true_value: torch.Tensor,
    device: torch.device,
    loss_function: TorchLoss,
    optimizer: Optimizer,
    decay: float,
    epochs: int,
    use_dynamic_graph: bool = True,
):
    clip = 5
    val_loss_list = []
    train_loss_list = []
    train_time = []
    rmse = []
    mae = []
    mape = []
    training_start_timestamp = str(datetime.now().strftime("%Y_%m_%d_%Hh%M"))
    print(f"training_start_timestamp: {training_start_timestamp}\n\n")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)

    save_model_checkpoint(
        model=model, epoch=0, training_start_timestamp=training_start_timestamp
    )
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for i, batch in enumerate(tqdm(train_loader, desc="Training", file=sys.stdout)):
            train_w, train_d, train_r, train_t, train_adj_r = batch
            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)

            # dynamic or static adjacency matrix
            if use_dynamic_graph:
                adj = train_adj_r.to(device)
            else:
                batch_size = train_adj_r.shape[0]
                adj = static_adj.repeat(batch_size, 1, 1).to(device)

            model.train()  # train pattern
            optimizer.zero_grad()  # grad to 0

            output, _, A = model(train_w, train_d, train_r, static_adj, adj)

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
            net=model,
            val_loader=val_loader,
            loss_function=loss_function,
            static_adj=static_adj,
            device=device,
            epoch=epoch,
            use_dynamic_graph=use_dynamic_graph,
        )
        val_loss_list.append(valid_loss)

        # evaluate the model on testing set
        rmse1, mae1, mape1 = evaluate(
            model, test_loader, true_value, static_adj, device, epoch
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
