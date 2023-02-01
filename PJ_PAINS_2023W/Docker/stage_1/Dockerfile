FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

RUN apt-get clean && apt-get update
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl && apt-get install -y vim && apt-get install -y git
RUN apt-get install -y tmux && apt install -y libgl1-mesa-glx

RUN apt-get update 
RUN pip install --upgrade pip 

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install matplotlib pyflann-py3 seaborn scikit-learn scipy pandas 
RUN pip install numpy jupyter wandb tqdm opencv-python==4.5.5.64 jupyterlab
RUN pip install -U black