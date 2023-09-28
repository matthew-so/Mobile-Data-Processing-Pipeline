#!/bin/bash

mb_pipe_path="/data/Mobile-Data-Processing-Pipeline"
parquet_path="/data/kaggle"
fingerspelling_path="/data/Fingerspelling"

if [ "$1" == "ls" ]; then
    sudo docker ps -a
    exit 0
fi

if [ "$2" == "" ]; then
    echo "Specify a container name when calling launch, run or rm"
    exit 0
fi

if [ "$1" == "launch" ]; then
    if [ "$2" == "fingerspelling" ]; then
        sudo docker run \
            -dit \
            -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline \
            -v "$parquet_path":/root/kaggle \
            -v "$fingerspelling_path":/root/Fingerspelling \
            --name $2 rohitsridhar91/asl_sign_recognizer:v1.1
    else
        sudo docker run \
            -dit \
            -v "$mb_pipe_path":/root/Mobile-Data-Processing-Pipeline \
            --name $2 rohitsridhar91/asl_sign_recognizer:v1.1
    fi
    
    if [ "$3" != "" ]; then
        sudo docker cp "$3" $2:/root
        echo "$3 copied to $2:/root"
    fi
elif [ "$1" == "run" ]; then
    sudo docker exec -it $2 /bin/bash
elif [ "$1" == "rm" ]; then
    sudo docker stop $2
    sudo docker rm $2
else
    echo "Specify either launch (create the container), run (resume the existing container), rm (remove existing container), or ls"
fi
