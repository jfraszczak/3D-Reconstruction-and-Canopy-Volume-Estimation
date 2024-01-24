FROM ros:humble-ros-base

# ********************************************************
# * Set your username and group names below *
# ********************************************************

ARG USERNAME=root
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# ********************************************************
# 
# ********************************************************

RUN apt-get update && apt-get install -y \
    python3-pip

# Create a non-root user
RUN groupadd -g $USER_GID $USERNAME \
	&& useradd -u $USER_UID -g $USER_GID -m $USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

#install dependencies
RUN pip3 install --user rosbags
 
USER root

USER $USERNAME

CMD ["bash"]
WORKDIR /workspace