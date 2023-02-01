docker build -t stage2 . 

docker run -it --gpus all --name stage2 \
        -p 9999:9999 \
        --ipc=host \
        -v /home/seunghun/바탕화면/PAINS_Football:/football \
        -v /home/seunghun/.vscode-server/data/Machine:/root/.vscode-server/data/Machine \
        stage2