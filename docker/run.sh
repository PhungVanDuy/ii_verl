sudo docker run --gpus all -it \
    -v $(pwd):/workspace/ii_verl \
    --workdir /workspace/ii_verl \
    --shm-size=8g \
    --network=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    verl