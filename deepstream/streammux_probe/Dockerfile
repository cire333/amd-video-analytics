FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev python3-gst-1.0 python3-pip python3-numpy libpython3.10 python3-opencv && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl || \
    echo "pyds wheel install failed"
