#!/bin/bash

mb_pipe_path="$HOME/Mobile-Data-Processing-Pipeline"
vimrc_path="$HOME/.vimrc"

if [ "$1" == "launch" ]; then
    sudo docker run -dit -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline --name sign_recognition gurudesh/copycat:copycat-gpu-cuda10.2-cudnn7
    sudo docker cp "$vimrc_path" sign_recognition:/root
elif [ "$1" == "launch_jup" ]; then
    sudo docker run -dit -p 8888:8888 -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline --name sign_recognition_jup gurudesh/copycat:copycat-gpu-cuda10.2-cudnn7
    sudo docker cp "$vimrc_path" sign_recognition_jup:/root
elif [ "$1" == "run_jup" ]; then
    sudo docker exec -it sign_recognition_jup /bin/bash
elif [ "$1" == "run" ]; then
    sudo docker exec -it sign_recognition /bin/bash
elif [ "$1" == "run_command" ]; then
    if [ "$2" == "" ]; then
        echo "Must pass the command as the second argument"
        exit 1
    fi
    sudo docker exec -d sign_recognition "$2"
else
    echo "Specify either launch (create the container) or run (resume the existing conainer)"
fi
