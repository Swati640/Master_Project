FROM vanessa/pytorch-dev:py2.7
WORKDIR /usr/src/app
COPY generate .
RUN python --version
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install numpy
RUN python -m pip install weave
RUN python -m pip install torch
RUN python -m pip install torchvision
RUN python -m pip install opencv-contrib-python-headless
RUN python -m pip install tqdm
RUN python -m pip install h5py
#RUN python -m pip install --no-cache-dir -r requirements.txt
CMD ["python", "generate_dataset.py"] 