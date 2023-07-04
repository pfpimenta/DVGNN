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


def save_training_report(
    training_start_timestamp: str,
    training_params: Dict[str, Any],
    training_stats: Dict[str, List],
    test_stats: Dict[str, float],
):
    """Saves training information in a JSON file
    (stats of each epoch, validation results).
    """
    training_report = {
        "training_start_timestamp": training_start_timestamp,
        "training_params": training_params,
        "training_stats": training_stats,
        "test_stats": test_stats,
    }
    folderpath = (
        Path(__file__).parent.parent.parent.resolve() / "data" / "Transportation"
    )
    json_filepath = folderpath / f"training_report_{training_start_timestamp}.json"
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
    txt_filepath = folderpath / f"training_report_{training_start_timestamp}.txt"
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
