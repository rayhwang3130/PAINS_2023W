PAINS 2022~2023 동계 프로젝트

# 패스를 통한 어시스트 상황에서 해당 패스의 득점 확률 변동 계산

## 부제 : PL 최고의 패스마스터 찾기 – 패스를 더 잘 줄 수는 없었는가?

팀원 : 조장) 황우현, 조원) 한승헌, 강민혁, 
---

## Bird directory tree

학습과 추론에 필요한 데이터는 삭제했습니다.
아래 링크에서 폴더를 다운로드 받은 뒤, 각 폴더/파일을 directory에 맞는 위치로 옮겨주세요.
https://works.do/568XHQX

```bash
.
│  arguments.py
│  main.py
│  new_main.py
│  README.md
│  requirements.txt
│  run.sh
│  tree.txt
│  utils2.py
│  Records.csv 
│ 
├─data
├─deep_sort_pytorch
│  ├─configs
│  │      deep_sort.yaml
│  │      
│  ├─deep_sort
│  │  │  deep_sort.py
│  │  │  __init__.py
│  │  │  
│  │  ├─deep
│  │  │      evaluate.py
│  │  │      feature_extractor.py
│  │  │      model.py
│  │  │      original_model.py
│  │  │      test.py
│  │  │      train.py
│  │  │      __init__.py
│  │  │      
│  │  └─sort
│  │          detection.py
│  │          iou_matching.py
│  │          kalman_filter.py
│  │          linear_assignment.py
│  │          nn_matching.py
│  │          preprocessing.py
│  │          track.py
│  │          tracker.py
│  │          __init__.py
│  │          
│  └─utils
│          asserts.py
│          draw.py
│          evaluation.py
│          io.py
│          json_logger.py
│          log.py
│          parser.py
│          tools.py
│          __init__.py
│          
├─elements
│      assets.py
│      deep_sort.py
│      perspective_transform.py
│      yolo.py
│      
├─image
├─inference
│  │  black.jpg
│  │  
│  ├─cropped
│  │  └─train
│  └─output
│      └─train
├─perspective_transform
│  ├─data
│  │      aligned_dataset.py
│  │      base_dataset.py
│  │      base_data_loader.py
│  │      custom_dataset_data_loader.py
│  │      data_loader.py
│  │      image_folder.py
│  │      single_dataset.py
│  │      two_aligned_dataset.py
│  │      unaligned_dataset.py
│  │      __init__.py
│  │      
│  ├─data_2
│  │      my_plot_field.m
│  │      read_me.py
│  │      
│  ├─deep
│  │      camera_dataset.py
│  │      contrastive_loss.py
│  │      siamese.py
│  │      
│  ├─homography
│  │      matrix.npy
│  │      
│  ├─models
│  │      base_model.py
│  │      cycle_gan_model.py
│  │      models.py
│  │      networks.py
│  │      pix2pix_model.py
│  │      test_model.py
│  │      two_pix2pix_model.py
│  │      __init__.py
│  │      
│  ├─pytorch-two-GAN-master
│  │      test.py
│  │      test_two_pix2pix.py
│  │      train.py
│  │      train_two_pix2pix.py
│  │      
│  └─util
│          image_pool.py
│          iou_util.py
│          projective_camera.py
│          rotation_util.py
│          synthetic_util.py
│          util.py
│          visualizer.py
│  
├─train
│        1.jpg
│        2.jpg
│         ...
│       605.jpg
│ 
├─processed_records
├─weights
└─yolov5
    ├─models
    │      common.py
    │      experimental.py
    │      export.py
    │      yolo.py
    │      __init__.py
    │      
    └─utils
            activations.py
            augmentation.py
            autoanchor.py
            datasets.py
            general.py
            google_utils.py
            loss.py
            metrics.py
            plots.py
            torch_utils.py
            __init__.py

```

python new_main.py \
    --df './Records.csv' \
    --df_name 'Processed_records' \
    --output_dir './inference/output/train/' \
    --crop_dir './inference/cropped/train/' \
    --black_dir './inference/black.jpg' \
    --img_dir './image/train/' \
    --weight './weights/yolov5s.pt' \
    --near_param 50

## Stage 2 directory tree

```bash
.
│  inference.py
│  inference.sh
│  log.py
│  main.py
│  Processed_records.csv
│  requirements.txt
│  train.py
│  train.sh
│  tree2.txt
│  
├─expected_xG
│      
├─inference
│  └─cropped
│      └─train
├─processed_records
├─trained_model
│  └─best
│      │  assets.json
│      │  config.yaml
│      │  data_processors.pkl
│      │  df_preprocessor.pkl
│      │  events.out.tfevents.1675082837.8994e2a9736b.61031.0
│      │  hparams.yaml
│      │  model.ckpt
│      │  
│      └─hf_text
│              config.json
│              special_tokens_map.json
│              tokenizer.json
│              tokenizer_config.json
│              vocab.txt
│              
└─util
        utils.py
```

모델 1차 학습
bash train.sh

학습된 모델을 이용한 추론
bash inference.sh
