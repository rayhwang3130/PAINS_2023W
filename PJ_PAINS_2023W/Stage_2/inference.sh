player_name='Arnold Kane KDB'

for pn in $player_name; do
    python inference.py \
        --input_dir './processed_records/'$pn'_processed.csv' \
        --near_length 50 \
        --weight_dir './trained_model/best/' \
        --save_dir './expected_xG/' \
        --label 'xG'
done
