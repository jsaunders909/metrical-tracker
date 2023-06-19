FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

RUN echo "conda activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

ENV CONDA_DEFAULT_ENV $(head -1 /tmp/environment.yml | cut -d' ' -f2)

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install gcc -y
RUN apt-get install --reinstall build-essential -y
RUN pip install insightface
RUN git config --global --add safe.directory /mnt/faster0/jrs68/metrical-tracker
RUN git config --global --add safe.directory /mnt/faster0/jrs68/metrical-tracker/MICA

RUN git config --global user.name Jack
RUN git config --global user.email jacksaunders909@gmail.com
