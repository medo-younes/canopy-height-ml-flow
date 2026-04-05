FROM continuumio/miniconda3

## Build environment
ADD environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml
RUN echo "conda activate canopy-flow" > ~/.bashrc
ENV PATH=/opt/conda/envs/canopy-flow/bin:$PATH
ENV METAFLOW_USER=canopy-flow

## Add required files
ADD ./flows/ /app/
ADD ./src/ /app/src
ADD ./.env /app/.env
ADD ./config.yaml /app/config.yaml
ADD ./credentials.json /app/credentials.json
WORKDIR /app/
