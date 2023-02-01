from util.utils import load_data
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
import os
import argparse
from tqdm import tqdm


def run(args):

    df = pd.read_csv(args.input_dir)
    features, _ = load_data(df, args.near_length)

    model = MultiModalPredictor(label=args.label, problem_type="regression")

    model = model.load(args.weight_dir)

    output = model.predict(features)

    new_df = pd.DataFrame(columns=["ball_x", "ball_y", "Real_xG", "Predicted_xG"])

    for i in tqdm(range(10, 60, 10)):
        for j in range(10, 60, 10):
            test = features.copy()
            test["ball_x"] = test["ball_x"].apply(lambda x: x + i)
            test["ball_y"] = test["ball_y"].apply(lambda x: x + j)
            xG = model.predict(test)
            temp = pd.DataFrame(
                {
                    "Ball_x": test["ball_x"],
                    "Ball_y": test["ball_y"],
                    "Real_xG": test["xG"],
                    "Predicted_xG": xG,
                }
            )
            new_df = pd.concat([new_df, temp])

    os.makedirs(args.save_dir, exist_ok=True)

    player_name = args.input_dir.split("/")[-1].split(".")[0] + "_xG"

    new_df.to_csv(args.save_dir + player_name + ".csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xG Prediction")
    parser.add_argument(
        "--input_dir",
        default="./processed_records/Arnold_processed.csv",
        type=str,
        help="Dataframe dir",
    )
    parser.add_argument(
        "--near_length",
        choices=[50, 100, 150, 200],
        default=100,
        type=int,
        help="Define Near_length",
    )
    parser.add_argument(
        "--weight_dir",
        default="./trained_model/",
        type=str,
        help="Define Model weight dir",
    )
    parser.add_argument(
        "--save_dir", default="./expected_xG/", type=str, help="Define result save dir"
    )
    parser.add_argument("--label", default="xG", type=str, help="Target label name")

    args = parser.parse_args()

    run(args)
