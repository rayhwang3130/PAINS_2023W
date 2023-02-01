import sys
from elements.yolo import YOLO
from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix
import pandas as pd
import torch
import os
import cv2
import numpy as np
from glob import glob
import argparse
from utils2 import *
from arguments import ArgumentsBase, Arguments
from pathlib import Path


def detect_img(detector, transform, img_dir, black_dir, output_dir, crop_dir):

    img_name = img_dir.split("/")[-1].split(".")[0]

    frame = cv2.imread(img_dir)
    h, w = frame.shape[0], frame.shape[1]
    output = detector.detect(frame)

    M, warped_image = transform.homography_matrix(frame)

    gt_img = cv2.imread(black_dir)
    gt_h, gt_w, _ = gt_img.shape
    bg_img = gt_img.copy()

    one = [x["label"] for x in output]
    ball_bool = "ball" in one
    coord_list = []

    yolopd = pd.DataFrame(output)
    print(yolopd)
    player_crop = np.array(yolopd.loc[yolopd["label"] == "player", "bbox"].tolist())
    player_crop = np.array(pd.DataFrame(output)["bbox"].tolist())
    left_x_end, left_y_up, right_x_end, right_y_down = (
        0,
        min(player_crop[:, 0, 1]) - 20,
        max(player_crop[:, 1, 0]),
        max(player_crop[:, 1, 1]),
    )
    cropped_img = frame[left_y_up:right_y_down, left_x_end:right_x_end, :]

    cv2.imwrite(crop_dir, cropped_img)

    if not ball_bool:

        print("No ball detected. Manually Detecting ball")

        mask = cv2.inRange(cropped_img, (0, 0, 0), (5, 5, 5))
        ball = np.mean(np.argwhere(mask == [255]), axis=0)
        x_center, y_center = int(ball[1] + left_x_end), int(ball[0] + left_y_up)

        ball_coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
        cv2.circle(bg_img, (ball_coords), 10, (102, 0, 102), -1)

    for i, obj in enumerate(output):

        xyxy = [
            obj["bbox"][0][0],
            obj["bbox"][0][1],
            obj["bbox"][1][0],
            obj["bbox"][1][1],
        ]
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = xyxy[3]

        if obj["label"] == "player":
            coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
            coord_list.append(coords)
            try:
                cv2.circle(bg_img, coords, 10, (255, 0, 0), -1)
            except:
                pass

        elif obj["label"] == "ball":
            print("Ball detected")

            if not (x_center > left_x_end and x_center < right_x_end) and not (
                y_center > left_y_up and y_center < right_y_down
            ):
                print("ball incorrectly marked")
                continue

            ball_coords = transform_matrix(
                M, (x_center, y_center), (h, w), (gt_h, gt_w)
            )
            cv2.circle(bg_img, ball_coords, 10, (102, 0, 102), -1)

    print("Saved to {}".format(output_dir))
    cv2.imwrite(output_dir, bg_img)

    # Get how many players are surrounding the ball
    dist = euclidean(coord_list, ball_coords)
    # num_near = near_player(dist,opt.near_param)

    # Distance and angle from the goalline
    goal_line_mid = (28, int(gt_h / 2))
    goal_dist, angle = angle_dist(ball_coords, goal_line_mid)
    ball = [ball_coords[0] - goal_line_mid[0], ball_coords[1] - goal_line_mid[1]]

    return goal_dist, dist, angle, ball


def run(opt):

    weight = opt.weight
    detector = YOLO(weight, 0.75, 0.75)
    transform = Perspective_Transform()

    df = pd.read_csv(opt.df)

    records = df.copy()

    records["input_dir"] = ""
    records["crop_dir"] = ""

    records["output_dir"] = "Failed"
    records["dist"], records["angle"], records["player_dist"] = None, None, None
    records.loc[:, "ball"] = None

    lists = glob(opt.img_dir + "*.png")

    count = 0

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.crop_dir, exist_ok=True)

    for idx, x in enumerate(lists):

        img_name = x.split("/")[-1].split(".")[0]
        cropped_dir = opt.crop_dir + img_name + "_match_cropped.png"
        output_dir = opt.output_dir + img_name + "_match_mapped.png"
        new_idx = int(img_name) - 2

        records.loc[new_idx, "input_dir"] = x

        try:
            goal_dist, dist, angle, ball = detect_img(
                detector, transform, x, opt.black_dir, output_dir, cropped_dir
            )
            (
                records.loc[new_idx, "dist"],
                records.at[new_idx, "player_dist"],
                records.loc[new_idx, "angle"],
            ) = (goal_dist, dist, angle)
            records.at[new_idx, "ball"] = ball
            records.loc[new_idx, "output_dir"] = output_dir
            records.loc[new_idx, "crop_dir"] = cropped_dir

            print(records.loc[new_idx, :])
            print("Successfully Processed")

        except:
            count += 1
            print("Resource not retrieved properly")

        print("{} out of {} images processed!".format(idx + 1, len(lists)))
        print("{} out of {} images are poorly collected".format(count, idx + 1))

    os.makedirs("./processed_records/", exist_ok=True)

    records.to_csv("./processed_records/" + opt.df_name + ".csv", index=False)

    print("saved to {}".format(opt.df_name))
    return records


if __name__ == "__main__":

    opt = Arguments()

    ##############Added By SeungHun ###########################################################

    opt = opt._parse()

    print(opt)

    records = run(opt)
