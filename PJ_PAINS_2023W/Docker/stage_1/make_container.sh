docker build -t stage1 . 

docker run -it --gpus all --name stage1 \
        -p 8888:8888 \
        --ipc=host \
        -v /home/seunghun/바탕화면/PAINS_Football:/football \
        -v /home/seunghun/.vscode-server/data/Machine:/root/.vscode-server/data/Machine \
        stage1