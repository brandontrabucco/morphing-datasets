FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

VOLUME /data
RUN mkdir -p /root/.mujoco

RUN apt-get update  -y
RUN apt-get install -y wget unzip git

RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O /root/.mujoco/mujoco200_linux.zip
RUN wget https://www.roboti.us/download/mjpro150_linux.zip -O /root/.mujoco/mjpro150.zip
RUN wget https://www.roboti.us/download/mjpro131_linux.zip -O /root/.mujoco/mjpro131.zip

RUN unzip /root/.mujoco/mujoco200_linux.zip -d /root/.mujoco/
RUN unzip /root/.mujoco/mjpro150.zip -d /root/.mujoco/
RUN unzip /root/.mujoco/mjpro131.zip -d /root/.mujoco/
RUN mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200

COPY mjkey.txt /root/.mujoco/mjkey.txt

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro131/bin

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin' >> /root/.bashrc

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O /root/anaconda3.sh
RUN bash /root/anaconda3.sh -b -p /root/anaconda3
RUN echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> /root/.bashrc

RUN apt-get install -y libosmesa-dev libglew-dev patchelf libglfw3-dev build-essential
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev

RUN git clone https://github.com/brandontrabucco/morphing-datasets.git /root/morphing-datasets
RUN /root/anaconda3/bin/conda env create -f /root/morphing-datasets/environment.yml
