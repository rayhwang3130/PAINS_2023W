# PJ_PAINS_2023W
PAINS 2022~2023 동계 프로젝트

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

## 패스를 통한 어시스트 상황에서 해당 패스의 득점 확률 변동 계산

### 부제 : PL 최고의 패스마스터 찾기 – 패스를 더 잘 줄 수는 없었는가?

## 목차

## 서론

### 1-1. 탐구 동기 :

- 덕배야, 패스를 그렇게 주면 어떡해?

### 1-2. 탐구 목적 :

- 패스가 어시스트가 되어 득점을 한 상황을 분석
- 득점 상황 별 득점 확률 모델링
- 패스를 받은 선수 및 공격진 선수들의 위치 및 상황 별 득점 확률 계산
- 해당 패스가 최적의 패스였는지 파악, 지표화를 통해 어시스트 선수의 패스 센스 및 판단력 파악

## 본론

### 2-1. 용어 정의

- 패스 : FBREF에서 표기한 SCA 1에서 Pass (Live)만 고려, Fouled, Dribble, Pass (Dead)는 실제로 패스한 선수보다 득점한 선수의 개인 기량이 더 높게 작용함을 고려
- 득점 확률 : FBREF에서 득점 상황 별 xG 값을 타겟으로, 득점자와 골의 각도, 거리, 주변 수비 숫자, 헤딩 혹은 일반적 슈팅 등 변수를 토대로 모델링하여 도출하는 지표

### 2-2. 데이터셋 소개

- 학습 단계 데이터 : 19/20 시즌 **모든** 경기에서 득점 상황 순간 캡쳐
- 캡쳐된 이미지를 전처리
    - 수비팀 (파란색), 공격팀 (빨간색) 으로 구별되게 메모장으로 칠해주기
    - 골키퍼(노란색)는 별도의 색깔로 칠해주기
    - 공 (검정색) 위치도 별도로
- 전처리된 3D 이미지를 Homography 소스를 사용하여 2D화 (tactical analysis image)
- 2D화된 이미지에서 앞서 설명한 변수들인 각도, 거리, 수비 숫자 등 추출, 정리
- 20/21 + 21/22 PL 어시스트 순위 최상위 3명의 어시스트 장면 캡쳐, 전처리
- 해당 상황들에서의 득점자 득점 확률 & 주변 공격진의 득점 확률을 모델을 통해 계산
- 어시스트 선수의 능력 지표 = 득점자 득점 확률 – 인근 선수의 득점 확률 중 최고치 -> 높을수록 좋은 판단, 낮을수록 나쁜 판단 or 득점자의 역량 차이 : 케이스 스터디 필요

### 2-3. 데이터 분석 및 정리

---

---

### 2-4. 모델링

### 2-5. 모델을 통한 어시스트 선수 분석

---

---

### 2-6. 케이스 스터디 -> 추후 2-5 완료 후 선수 경기 질적 분석

<aside>
💡 2-1~3 : Phase 1 // 2-4 ~5 : Phase 2 // 2-6 : Phase 3 로 간소화

</aside>

## 결론

### 3-1. 탐구 요약 및 정리

### 3-2. 한계점

1. 부록

4-1. 참고문헌

4-2. R 코드
