near_length='50 100 150 200'
split_ratio='0.7 0.8 0.9'

for nl in $near_length; do
    for sr in $split_ratio; do
        echo "near_length: $nl, Split_ratio: $sr"
        save_name="Near_length_$nl-Split_ratio_$sr"
        python main.py \
            --input_dir './processed_records/Processed_records.csv' \
            --near_length $nl \
            --split_ratio $sr \
            --save_name $save_name
    done
done
