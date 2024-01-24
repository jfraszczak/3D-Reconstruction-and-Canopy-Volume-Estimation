FROM huggingface/transformers-pytorch-gpu:4.29.1

# ********************************************************
# * Set your username and group names below *
# ********************************************************

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# ********************************************************
#
# ********************************************************

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
build-essential ca-certificates python3.8 python3.8-dev python3.8-distutils git wget cmake

RUN apt-get update && \
apt install -y tmux

RUN ln -sv /usr/bin/python3.8 /usr/bin/python

# Create a non-root user
RUN groupadd -g $USER_GID $USERNAME \
	&& useradd -u $USER_UID -g $USER_GID -m $USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH=/workspace

RUN wget https://bootstrap.pypa.io/get-pip.py && \
python get-pip.py --user && \
 rm get-pip.py


# install dependencies
RUN pip install --user matplotlib
RUN pip install --user numpy
RUN pip install --user opencv-python==4.2.0.32
RUN pip install --user argparse


RUN pip install --user Pillow
RUN pip install --user pycocotools
RUN pip install --user torch
RUN pip install --user torchvision
RUN pip install --user transformers
RUN pip install --user evaluate
RUN pip install --user omegaconf
RUN pip install --user hydra-core

CMD ["bash"]
WORKDIR /workspace