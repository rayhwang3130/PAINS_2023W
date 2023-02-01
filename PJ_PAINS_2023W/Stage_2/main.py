from util.utils import load_data
from train import train
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
import os
import wandb
import argparse


def run(args):

    os.makedirs("./inference/", exist_ok=True)

    df = pd.read_csv(args.input_dir)

    df_rmna, hyperparameters = load_data(df, args.near_length)
    train_df = df_rmna[: int(len(df_rmna) * args.split_ratio)]
    test_df = df_rmna[int(len(df_rmna) * args.split_ratio) :]

    model = MultiModalPredictor(label=args.label, problem_type="regression")

    wandb.init(name=args.save_name, project="PAINS-Project", config=args)

    model = train(
        model=model,
        train_data=train_df,
        hyperparameters=hyperparameters,
        seed=args.seed,
    )

    save_dir = args.default_dir + args.save_name

    os.makedirs(save_dir, exist_ok=True)

    model.save(save_dir)

    scores = model.evaluate(df_rmna)
    wandb.log(scores)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAINS Project")

    parser.add_argument(
        "--input_dir",
        default="./processed_records/Processed_records.csv",
        type=str,
        help="Dataframe dir",
    )
    parser.add_argument(
        "--near_length",
        choices=[50, 100, 150, 200],
        type=int,
        help="Define Near_length",
    )

    parser.add_argument(
        "--split_ratio", choices=[0.7, 0.8, 0.9], type=float, help="Define Model name"
    )
    parser.add_argument(
        "--label", default="xG", type=str, help="Define name of target label"
    )
    parser.add_argument("--seed", default=1105, type=int, help="Define Seed Num")

    parser.add_argument(
        "--default_dir",
        default="./trained_model/",
        type=str,
        help="Define Model save dir",
    )
    parser.add_argument(
        "--save_name",
        default="Split_ratio-0.7_Near_length-50/",
        type=str,
        help="Define Model name",
    )

    args = parser.parse_args()

    run(args)
