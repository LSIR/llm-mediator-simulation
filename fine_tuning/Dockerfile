# Use CUDA base image with Python 3.12
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3.12
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir hatch

# Stage for RCP RunAI
FROM base as runai

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}
#####################################

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}
# Copy project files
COPY pyproject.toml /home/${LDAP_USERNAME}/
COPY train_olmo_reddit.py /home/${LDAP_USERNAME}/
COPY configs/ /home/${LDAP_USERNAME}/configs/
RUN mkdir -p /home/${LDAP_USERNAME}/data
COPY data/ /home/${LDAP_USERNAME}/data/

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Set working directory
WORKDIR /home/${LDAP_USERNAME}

# Install the project and its dependencies as root
RUN pip install --no-cache-dir -e .

# Switch to user
USER ${LDAP_USERNAME}

# Set the entrypoint
CMD ["accelerate", "launch", "/home/${LDAP_USERNAME}/train_olmo_reddit.py", "--cfg.training.output_path=/home/${LDAP_USERNAME}/olmo2-cga-cmv/sft"]

# Stage for local development (lsirpc36)
FROM base as local

# Set working directory
WORKDIR /fine-tuning

# Copy project files
COPY pyproject.toml .
COPY train_olmo_reddit.py .
COPY configs/ ./configs/
RUN mkdir -p ./data
COPY data/ ./data/
RUN mkdir -p ./output

# Install the project and its dependencies
RUN pip install --no-cache-dir -e .

# Set the entrypoint
CMD ["accelerate", "launch", "train_olmo_reddit.py", "training.output_path=output"]