# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def get_project_root_folderpath() -> str:
    project_folderpath = Path(__file__).parent.parent.parent.resolve()
    return project_folderpath


def get_training_results_dir(epoch: int, training_start_timestamp: str) -> str:
    project_folderpath = get_project_root_folderpath()
    training_results_dir = (
        project_folderpath
        / "training_results"
        / f"{training_start_timestamp}"
        / f"{epoch}"
    )
    training_results_dir.mkdir(parents=True, exist_ok=True)
    return training_results_dir


# def get_checkpoint_name(epoch: int, training_start_timestamp: str) -> str:
#     checkpoint_name = f"{training_start_timestamp}_e{epoch:0>3d}"
#     return checkpoint_name

# def get_training_results_dir(epoch: int, training_start_timestamp: str) -> str:
#     project_root_folderpath = get_project_root_folderpath()
#     checkpoint_name = get_checkpoint_name(epoch=epoch, training_start_timestamp=training_start_timestamp)
#     training_results_dir =
#     return training_results_dir


def save_training_report(
    training_start_timestamp: str,
    training_stats: Dict[str, List],
    # train_loss_list: List[float],
    # val_loss_list: List[float],
    # train_time_list: List[float],
    # rmse_list: List[float],
    # mae_list: List[float],
    # mape_list: List[float],
    test_stats: Dict[str, float],
):
    training_report = {
        "training_start_timestamp": training_start_timestamp,
        "training_stats": training_stats,
        "test_stats": test_stats,
    }
    folderpath = (
        Path(__file__).parent.parent.parent.resolve() / "data" / "Transportation"
    )
    json_filepath = folderpath / f"training_report_{training_start_timestamp}.json"
    with open(json_filepath, "w") as f:
        json.dump(training_report, f)

    print(f"Saved training stats on {json_filepath}")


def save_model_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    training_start_timestamp: str,
    # evaluation_overview: str,
    # performance_dict: Dict[str, Any],
    # training_loss_list: np.ndarray,
) -> None:
    """TODO description"""
    # saves model at training_results/{timestamp}/{epoch}/model.pt
    training_results_dir = get_training_results_dir(
        epoch=epoch, training_start_timestamp=training_start_timestamp
    )
    trained_model_filepath = training_results_dir / "model.pt"
    torch.save(model.state_dict(), trained_model_filepath)
    print(f"Saved {trained_model_filepath}")
    # TODO save other infos as well
    # prepare training information JSON
    # checkpoint_training_report = model.training_report
    # checkpoint_training_report["epoch"] = epoch
    # checkpoint_training_report["step"] = step
    # checkpoint_training_report["checkpoint_time"] = datetime.now().strftime(
    #     "%Y_%m_%d_%Hh%M"
    # )
    # save training information JSON
    # report_json_filepath = checkpoint_dir / "training_report.json"
    # save_dict_to_json(
    #     dict=checkpoint_training_report, json_filepath=report_json_filepath
    # )
    # print(f"Saved {report_json_filepath}")
    # save performance JSON
    # performance_json_filepath = checkpoint_dir / "performance.json"
    # save_dict_to_json(dict=performance_dict, json_filepath=performance_json_filepath)
    # print(f"Saved {performance_json_filepath}")
    # save evaluation overview
    # eval_overview_filepath = checkpoint_dir / "eval_overview.md"
    # with open(eval_overview_filepath, "w") as f:
    #     f.write(evaluation_overview)
    # print(f"Saved {eval_overview_filepath}")
    # # save training loss list
    # training_loss_list_filepath = checkpoint_dir / "training_loss_list.npy"
    # with open(training_loss_list_filepath, "wb") as file:
    #     np.save(file, training_loss_list)
    # print(f"Saved {training_loss_list_filepath}")
