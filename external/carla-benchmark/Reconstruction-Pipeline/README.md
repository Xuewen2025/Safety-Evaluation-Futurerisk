# CARLA Scenario Generation Pipeline

This document provides instructions on how to set up and run the CARLA Scenario Generation Pipeline using the provided Docker environment.

## Overview

The project is designed to generate, run, and validate driving scenarios in the [CARLA Simulator](https://carla.org/). It uses a containerized environment to ensure all dependencies and configurations are consistent. The pipeline includes various scripts for processing map data (from OpenStreetMap), generating scenes, managing scenarios, and interacting with the CARLA simulator.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed and configured on your host machine:

1.  **Docker Engine:** [Installation Guide](https://docs.docker.com/engine/install/)
2.  **NVIDIA GPU:** A dedicated NVIDIA graphics card is required.
3.  **NVIDIA Graphics Drivers:** Ensure you have the latest proprietary NVIDIA drivers for your GPU.
4.  **NVIDIA Container Toolkit:** This is necessary to provide the Docker container with access to the host's GPU. [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
5.  **An X11 Server:** The container forwards a GUI, so you must be running on a system with an X server.
6.  **GEMINI API KEY:** Gemini API Key to invoke the LLM integrated in the pipeline. 

### Build

First, you need to clone the repository and navigate to the `Reconstruction-Pipeline` directory.

```sh
git clone git@github.com:NahianSalsabil/carla-benchmark.git
cd Reconstruction-Pipeline
```

If you run `ls -F` command, then you can see the following files.
```sh
build_docker.sh  data/  Dockerfile  lib/  README.md  requirements.txt  run_docker.sh  scripts/
```

The Docker image contains all the necessary dependencies, including Python, `osmium-tool`, and the required Python packages.

To build the Docker image, follow the commands below. This is will create an image tagged as `carlatest:latest`.
```sh
docker build -t carlatest:latest .
```


### Usage (Docker)

To the run the newly created Docker image called `carlatest:latest`, follow the commands below to launch a Docker container.

First, you should allow X11 connections. On your host machine, enter the following in the terminal:
```sh
xhost +local:root
```

Then create and run a Docker container:
While running this command, you need to put the Gemini API Key in the designated field.

```sh

docker run -it \
    --privileged \
    --rm \
    --gpus all \
    --net=host \
    -e GEMINI_API_KEY="apikey" \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json" \
    -v "$(pwd)/../Benchmark:/home/carla/PythonAPI/util/Benchmark" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
    carlatest:latest
```

This will drop you into an interactive `bash` shell inside the container, in the `/home/carla` directory. The `docker run` command contains several important flags:
-   `--gpus all`: Provides the container with access to all available host GPUs.
-   `--net=host`: Shares the host's network stack with the container. This is required for the CARLA server and clients to communicate.
-   `-e DISPLAY=$DISPLAY` & `-v /tmp/.X11-unix...`: Forwards the host's display, allowing you to view GUI applications running inside the container (e.g., the CARLA spectator view).
-   `-e GEMINI_API_KEY=<put-key-here>`: Passes the Gemini API key to the container.
-   Other flags (`--privileged`, `-e NVIDIA_DRIVER_CAPABILITIES`, etc.) are set to ensure proper functioning of the graphics drivers within the container.


## Debugging

Since Carla Engine relies on GPU for graphical rendering, users may face some problem while trying to run the Docker container. For this reason, some additional tools (`nvidia-smi`, `vulkaninfo`) have been provided in the Docker image.

However, the Docker container may fail to load some necessary shared objects for GPU drivers. In that case, open your host machine terminal and enter the following:
```sh
find /usr/lib -name "libnvidia-gpucomp.so.*"
```

This will list the shared object files. Then you should copy those to the `lib/` directory and build the Docker image. The `Dockerfile` is set up in such a way that these shared objects are available in the `LD_LIBRARY_PATH` environment variable.

## Quick Pipeline Demo

This pipeline takes crash reports as input and generate maps and scenarios and run the simulation in CARLA. Some sample reports have alrady been provided in `/home/carla/PythonAPI/util/data/reports` directory so that you can directly run the pipeline without downloading the reports.

Once inside the container's shell, you can execute the various pipeline scripts. To run the pipeline, move to `/home/carla/PythonAPI/util/` within the container.

```bash
# Navigate to the directory containing the scripts
cd /home/carla/PythonAPI/util/
```

A bash file is already created to run the whole pipeline. Once you are in `/home/carla/PythonAPI/util/`, run

```bash
# Generates the maps and invokes the LLM to get the vehicles' initial state for all the reports provided in the `/home/carla/PythonAPI/util/data/reports` directory
./generate_map.sh
```

Once you are done with this step, you can run the following command with the specific crash id to run the simulation in carla.

```bash
# Generates the scene, opens CARLA Engine, load the map and simulate the vehicles. Also saves the trajectory and vehicles' state for future use.
./generate_scene.sh <crash_id>
```

If you want to simulate the already created scene again, run
```bash
# loads the previously created scene and run in carla simulator.
./run_scenario.sh <crash_id>
```


## Usage (Pipeline)

**Important: the Carla version 0.9.15 explicitly relies on Python version 3.8.** Which is why you should use the packaged `python3.8` command to run the Python scripts. 
Once inside the container's shell, you can execute the various pipeline scripts. To run the pipeline, move to `/home/carla/PythonAPI/util/` within the container.

```bash
# Navigate to the directory containing the scripts
cd /home/carla/PythonAPI/util/
```

To run the pipeline from the crash reports, You need to download the reports first from Get Crash Details API of [NHTSA Crash Viewer](https://crashviewer.nhtsa.dot.gov/CrashAPI)

The sample API is: 

https://crashviewer.nhtsa.dot.gov/crashviewer/CrashAPI/crashes/GetCaseDetails?stateCase=510003&caseYear=2023&state=51&format=xml

You need to provide the `stateCase` Number, `caseYear` and the `state` number in the designated place to view that specific crash case.
Before working with the NHTSA Crash Viewer, I would suggest to give some time to understand how the APIs work.

After downloading the reports, put all the crash reports in the `/home/carla/PythonAPI/util/data/reports` directory. When putting the reports in this directory, you must follow the naming convension "crash_<crash_number>.xml". Some sample reports have alrady been provided in this directory so that you can take a look at those.

Follow the same process from [Quick Pipeline Demo](#quick-pipeline-demo) to run the pipeline with newly downloaded crash reports.

If you like, please refer to the bash scripts to understand the execution order of the scripts.


## Prompt Templates

The `scripts/prompt/` directory contains the prompt templates used by the LLM stage of the reconstruction pipeline. It consists of two files:

*   **`directives.txt`**: The unified prompt template. It contains the static role definition, critical traffic rules, required reasoning steps, and the expected JSON output schema — everything that stays constant across all accident reconstructions. Per-accident data is injected via a designated placeholder.
*   **`input_data.txt`**: The per-accident variable. It provides two inputs substituted into `directives.txt` at runtime:
    *   **Crash Summary** — the extracted narrative text of the accident.
    *   **Map Context (Local CARLA Coordinates)** — one entry per vehicle, generated by `ns_extract_road.py` from the `.xodr` map. Each entry contains:

    | Field | Description |
    | :--- | :--- |
    | `road_id` | OpenDRIVE ID of the road segment. |
    | `segment` | Four `[x, y]` corner points bounding the road segment area. The LLM selects the box containing the vehicle's pre-crash start position. |

These templates are not required to run pre-generated benchmarks, but they document the prompts used to create the benchmark and provide the templates needed to reconstruct additional scenarios.

## Working with the Pipeline without Docker

If you do not want to use the docker and want to use the scripts in your own CARLA repository, you need to put all the `.py` scripts in `your_carla_root/PythonAPI/util` and you are good to go.



