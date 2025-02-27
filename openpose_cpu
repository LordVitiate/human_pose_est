# Use the official PyTorch image with CUDA 9.0 as a base
FROM pytorch/pytorch:0.4.0-cuda9.0-cudnn7-runtime

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a new conda environment with Python 3.6
RUN conda create -n myenv python=3.6 -y && \
    conda clean -afy

# Install necessary packages in the conda environment
RUN /opt/conda/bin/conda install -n myenv numpy scipy matplotlib && \
    conda clean -afy

# Upgrade pip and install additional packages using a shell script
RUN echo "source activate myenv && \
    pip install --upgrade pip && \
    pip install cython setuptools tqdm==4.66.2 && \
    pip install -e git+https://github.com/ncullen93/nitrain.git#egg=nitrain && \
    pip install visdom nibabel" > /tmp/install_packages.sh && \
    chmod +x /tmp/install_packages.sh && \
    /bin/bash /tmp/install_packages.sh

# Set the default command to bash
CMD ["/bin/bash"]