python new_main.py \
    --df './Records.csv' \
    --df_name 'Processed_records' \
    --output_dir './inference/output/train/' \
    --crop_dir './inference/cropped/train/' \
    --black_dir './inference/black.jpg' \
    --img_dir './image/train/' \
    --weight './weights/yolov5s.pt' \
    --near_param 50