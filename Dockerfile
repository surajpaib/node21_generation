FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y libxext6 libglib2.0-0 libsm6 build-essential sudo \
    libgl1-mesa-glx git wget rsync tmux nano dcmtk fftw3-dev liblapacke-dev libpng-dev libopenblas-dev jq && \
  rm -rf /var/lib/apt/lists/*

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm nodule_patches /opt/algorithm/nodule_patches

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm ct_nodules.csv /opt/algorithm/

RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# Copy model files to the docker
COPY --chown=algorithm:algorithm model /opt/algorithm/model

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=nodulegeneration

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=40G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=11G


