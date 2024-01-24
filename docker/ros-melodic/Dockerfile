FROM ros:melodic-ros-base-bionic

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
	build-essential ca-certificates git wget sudo cmake
# RUN ln -sv /usr/bin/python2.7 /usr/bin/python
RUN apt install -y python-pip

# Create a non-root user
RUN groupadd -g $USER_GID $USERNAME \
	&& useradd -u $USER_UID -g $USER_GID -m $USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

#install dependencies
RUN pip install --user matplotlib
RUN pip install --user numpy
RUN pip install --user opencv-python==4.2.0.32
RUN pip install --user argparse

USER root

RUN apt-get update
RUN apt install -y ros-melodic-cv-bridge
RUN apt install -y net-tools
RUN apt install -y x11-apps
RUN apt install -y iputils-ping
RUN apt-get install -y xauth
RUN apt-get install -y ros-melodic-rviz
# RUN wget https://github.com/foxglove/studio/releases/download/v1.55.0/foxglove-studio-1.55.0-linux-amd64.deb
# RUN apt install -y ./foxglove-studio-*.deb
RUN apt update && sudo apt install -y foxglove-studio

RUN apt install -y ros-melodic-rosbridge-suite

USER $USERNAME

CMD ["bash"]
WORKDIR /workspace