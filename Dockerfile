FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

Expose 6006 6007 6008 6009

Run apt-get update && apt-get install -y \
    software-properties-common
Run add-apt-repository universe
Run apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    libjpeg-dev \
    libpng-dev 

Run pip3 install --upgrade pip
Run pip3 uninstall tensorboard -y \
		   nvidia-tensorboard -y \
		   jupyter-tensorboard -y \
		   tensorboard-plugin-wit -y \
		   tensorboard-plugin-dlprof -y
Run pip3 install ffmpeg
Run pip3 install matplotlib
Run pip3 install prefetch_generator
Run pip3 install librosa==0.10.1
Run pip3 install omegaconf==2.0.6
Run pip3 install pytorch_lightning==1.2.10

Run ldconfig && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* /tmp/*

RUN mkdir /root/nuwave2
WORKDIR /root/nuwave2

COPY *py /root/nuwave2/
COPY *yaml /root/nuwave2/
COPY utils /root/nuwave2/utils
COPY train_vast.ipynb /root/nuwave2/
