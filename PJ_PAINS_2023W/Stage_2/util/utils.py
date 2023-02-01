import pandas as pd
import re
import numpy as np
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import os


def load_data(df: pd.DataFrame = None, near_length: int = 100):

    df_rmna = df.loc[
        df["player_dist"].notna(),
        ["xG", "Distance", "Body Part", "crop_dir", "angle", "player_dist", "ball"],
    ].copy()

    # Change string list to int list
    try:
        df_rmna["player_dist"] = df_rmna["player_dist"].apply(
            lambda x: list(map(int, re.findall(r"\d+", x)))
        )
        df_rmna["ball"] = df_rmna["ball"].apply(
            lambda x: list(map(int, re.findall(r"\d+", x)))
        )
    except:
        pass

    # Extract how many players are surrounding a player whose possessing the ball
    df_rmna["player_dist"] = [
        sum(np.array(x) < np.array([near_length])) for x in df_rmna["player_dist"]
    ]

    # Extract Ball coordinate
    df_rmna["ball_x"] = df_rmna["ball"].apply(lambda x: x[0])
    df_rmna["ball_y"] = df_rmna["ball"].apply(lambda x: x[1])

    df_rmna.drop(columns="ball", inplace=True)

    # Header, Shoot encoding
    df_rmna["Goal_code"] = df_rmna["Body Part"].apply(lambda x: int("Foot" in x))

    df_rmna.drop(columns="Body Part", inplace=True)

    hyperparameters = get_hyperparameter_config("multimodal")

    return df_rmna, hyperparameters
