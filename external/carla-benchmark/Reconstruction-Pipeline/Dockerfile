# Use the official CARLA 0.9.15 image as base
FROM carlasim/carla:0.9.15

USER root

RUN chown -R carla:carla /home/carla

COPY --chown=carla:carla . /home/carla/app

RUN mv /home/carla/app/scripts/* /home/carla/PythonAPI/util/ \
    && mv /home/carla/app/data /home/carla/PythonAPI/util/ 
    
RUN ln -s /home/carla/PythonAPI/carla/agents /home/carla/PythonAPI/util/agents

RUN apt-get update && apt-get install -y \
    wget \
    osmium-tool \
    xdg-user-dirs \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-utils \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# installing Python 3.8 for Carla 0.9.15
RUN apt-get update && apt-get install -y gnupg ca-certificates curl \
    && curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776' \
    -o /etc/apt/trusted.gpg.d/deadsnakes.asc \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu bionic main" \
    > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update \
    && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils

# installing pip for Python 3.8
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py \
    && python3.8 get-pip.py \
    && rm get-pip.py 

ENV PATH="$PATH:/home/carla/.local/bin:/home/carla/PythonAPI/carla"

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/carla/app/lib"

WORKDIR /home/carla

RUN pip3.8 install -r app/requirements.txt

USER carla

# Define the entrypoint
CMD ["/bin/bash"]

