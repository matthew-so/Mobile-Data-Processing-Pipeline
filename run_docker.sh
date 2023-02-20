#!/bin/bash

mb_pipe_path="/data/Mobile-Data-Processing-Pipeline"
# vimrc_path="$HOME/.vimrc"

if [ "$1" == "launch" ]; then
    sudo docker run -dit -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline --name sign_recognition rohitsridhar91/asl_sign_recognizer:v1.1
    # sudo docker run -dit -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline --name sign_recognition gurudesh/copycat:copycat-gpu-cuda10.2-cudnn7
    # sudo docker run -dit -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline --name sign_recognition gurudesh/copycat:copycat-cpu-latest-no-gt2k
    if [ "$2" != "" ]; then
        sudo docker cp "$2" sign_recognition:/root
        echo "$2 copied to sign_recognition:/root"
    fi
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
