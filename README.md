# Ultrack's CTC submission

This repository contains the code to reproduce the results of the Ultrack submission to the Cell Tracking Challenge.

For more information about the challenge see https://celltrackingchallenge.net

For more information about Ultrack see https://github.com/royerlab/ultrack

## Requirements

- Docker: https://docs.docker.com/get-docker/
- NVIDIA GPU with at least 12 GB of RAM. Execution tested on 24 GB.
- Gurobi Web License Service (WLS) because it's running in a docker container. See https://github.com/royerlab/ultrack/tree/main/docker#gurobi-support for additional instructions.

## Setup

1. Clone the repository
    ```bash
    git clone https://github.com/royerlab/ultrack_CTC_submission
    ```

2. Change to the repository directory
    ```bash
    cd ultrack_CTC_submission
    ```

3. Download pre-trained weights
    ```bash
    bash download.sh
    ```

4. Build docker image
    ```bash
    docker build -t ultrack_ctc .
    ```

## Reproducing CTC submission

Once setup is complete, you can run the experiments by running the docker container. 

```bash
docker run --rm -it --gpus all \
    -v weights/:/app/weights \
    -v <YOUR_DATA_DIR>:/app/data \
    -v <LARGE_DATA_STORAGE>:/wkdir \
    -e WK_DIR=/wkdir \
    -v <PATH TO YOUR GUROBI WSL LICENSE>:/opt/gurobi/gurobi.lic \
    ultrack-ctc
```

The variable `$WK_DIR` and  `/wkdir` are useful to store intermediate results, for the TRIF dataset this can be more than 500GBs, which otherwise will be saved in your main storage unit.

## General usage

For general usage and application of our tracking algorithm on your own data see https://github.com/royerlab/ultrack.


## Additional notes

To train your own model, see `train.sh` and `train.py`.
This requires at least 24 GB of GPU memory depending on the dataset.
