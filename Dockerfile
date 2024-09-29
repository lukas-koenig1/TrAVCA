# Wähle das offizielle Ubuntu-Image als Basis
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Setze eine nicht-interaktive Umgebungsvariable, um Probleme bei der Installation von Paketen zu vermeiden
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"

# Update und installiere notwendige Pakete
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installiere Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/miniconda && \
    rm /miniconda.sh

# Füge Conda zu PATH hinzu
ENV PATH=/opt/miniconda/bin:$PATH

# Kopiere die environment.yml in das Docker-Image
COPY environment.yml .

# Erstelle die Conda-Umgebung basierend auf der environment.yml
RUN conda env create -f environment.yml

# Aktiviere die Conda-Umgebung und setze sie als Standard
RUN echo "source activate $(head -n 1 environment.yml | cut -d ' ' -f 2)" > ~/.bashrc
ENV PATH /opt/miniconda/envs/$(head -n 1 environment.yml | cut -d ' ' -f 2)/bin:$PATH

# Set Cuda
#RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
#RUN conda install -n BA pytorch==2.3.1 torchvision==0.15.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub
# RUN apt-get update

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-libraries-12-2 \
#     cuda-libraries-dev-12-2 \
#     cuda-tools-12-2

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-toolkit-12-2

# CUDA-Pfad hinzufügen
ENV PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}


# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Exponiere den Jupyter-Port
EXPOSE 8888

# Füge den Befehl hinzu, um den Jupyter Notebook Server zu starten
CMD ["bash", "-c", "source activate TrAVCA && jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"]

# Build command:
# docker build -t travca .

# Run commmand:
# docker run --gpus all --shm-size=4gb -p 8888:8888 -v ${pwd}:/app -it travca