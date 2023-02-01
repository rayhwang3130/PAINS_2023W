FROM autogluon/autogluon:0.6.1-cuda11.3-jupyter-ubuntu20.04-py3.8

RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

RUN apt-get clean && apt-get update
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl && apt-get install -y vim && apt-get install -y git
RUN apt-get install -y tmux && apt install -y libgl1-mesa-glx

RUN apt-get update 
RUN pip install --upgrade pip 

RUN pip install wandb pandas