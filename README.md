# CognitiveEMS-Trial-Task
UVA DSA - CognitiveEMS - Undergraduate Research Trial Task

1. Overview
    The goal of this project was to modify the EgoEMS simulator client so that it preforms object detection on video data streamed from the EgoEMS server.


2. Prerequisites
    Tested on Ubuntu 26.04 with NVIDIA RTX 5060 Laptop GPU.
    Docker and the NVIDIA Container Toolkit are required for using the DETR server.
    I used a Conda environment for the simulator server and client.
    
    To create and activate the Conda environment download Conda (Miniconda or Anaconda) and run
    ```bash
    conda create -n egoems_sim python=3.11
    conda activate egoems_sim
    ```
    You must activate the conda environment (second command) everytime you wish to run the simulator server and client. Make sure you are inside the "egoems_sim" conda environment before installing the dependencies bellow.
    
    Simulator Server Dependencies:
        The system level dependencies "ffmpeg" and "libsrt" and python packages "PyQt5"  are required to run the server GUI. To install, run
        ```bash
        sudo apt update
        sudo apt install ffmpeg libsrt-openssl-dev
        pip install PyQt5
         ```        

    Simulator Client Dependencies:
        Python packages "numpy", "requests", and "opencv" are required to run the client. To install, run
        ```bash
        pip install numpy requests opencv-python
        ```

    Initializing the Simulator Submodule:
        To initialize the simulator repository (which is included as a submodule in this repository), run 
        ```bash
        git submodule update --init --recursive
        ```
    
    DETR Server:
        The prebuilt TensorRT engine is not be compatible with all GPU architectures. On my RTX 5060 Laptop GPU, I repbuilt the engine locally using the EMS-Pipeline repository and DETR checkpoint. Follow the directions at 
        "https://github.com/UVA-DSA/EMS-Pipeline/blob/demo_2026/Tools/EMS_Vision/README_container_inference.md" under the header "Build from Source" if the egine is not compatible with your GPU. Otherwise, follow the following set
        up instructions.

        To verify Docker and GPU runtime integration run
        ```bash
        docker --version
        docker run --rm hello-world
        docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
        ```
        All commands should be successful.
        
        Next, pull the prebuilt image and start the inference server:
        ```bash
        docker login
        docker pull keshara2032/egoems-inference-server:latest
        docker run --gpus all --rm -d \
            --name egoems-inference-server \
            -p 8000:8000 \
            -e ACTIVITY_ENGINE_PATH= \
            -e ACTIVITY_FEATURE_ENGINE_PATH= \
            keshara2032/egoems-inference-server:latest
        ```
        
3. Running the Client:
    To run the simulation server, run the following commands from inside the EgoEMS-Sim submodule:
    ```bash
    conda activate egoems_sim
    python simulator/server/gui_server.py
    ```
    Using the UI, specify a egocentric video and csv file to use. By default, Port should be set to 9000, FPS to 30, Width to 480, and Height to 270. Leave this unchanged. Finally, click "Start".

    To run the object detection client, run the following command from a seperate terminal inside the cognitive_ems_trial_jaylbeck repository root:
    ```bash
    conda activate egoems_sim
    python src/detr_stream_client.py 127.0.0.1 --port 9000 --width 480 --height 270
    ```
    The client sends every 5th frame to the DETR server for object detection. It then generates an image displaying model detections and stores it in "src".
    
