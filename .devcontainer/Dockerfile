FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG UID
ARG GID

ENV UID=${UID:-1000}
ENV GID=${GID:-1000}

RUN pip install --upgrade pip

RUN pip install flake8 \
    pep8-naming \
    black \
    isort \
    ipykernel

RUN apt-get update && apt install -y software-properties-common

# install fish shell
RUN apt-add-repository ppa:fish-shell/release-3 -y && \
    apt-get update && \
    apt-get install -y fish

# create user jupyter (uid 1000, gid 1000)
RUN addgroup --system --gid ${GID} jupyter && \
    adduser jupyter --system --uid ${UID} --gid ${GID} --shell /usr/bin/fish

RUN apt-get install -y curl git sudo gcc g++ swig

RUN sudo apt-get install -y unzip dnsutils iputils-ping

# add user to sudoers
RUN echo "jupyter ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER jupyter

# install oh-my-fish
RUN curl -L https://get.oh-my.fish > /tmp/install && \
    fish /tmp/install --noninteractive --path=~/.local/share/omf --config=~/.config/omf && \
    rm /tmp/install

# install agnoster theme
RUN fish -c "omf install agnoster" && \
    fish -c "omf theme agnoster"

# change default shell to fish
SHELL ["/usr/bin/fish", "-c"]

RUN /opt/conda/bin/pip install "pydantic<2.0.0" lightning scikit-learn matplotlib

RUN /opt/conda/bin/pip install torchsampler wandb

RUN pip install ta stable-baselines3[extra]

RUN pip install numpy pandas

RUN pip install gluonts transformers

RUN pip install accelerate evaluate

RUN pip install reformer_pytorch

RUN pip install datasets ujson

RUN sudo apt-get update && sudo apt-get install -y \
    build-essential wget

RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h && \
    ./configure --prefix=/usr/local && \
    make && \
    sudo make install && \
    sudo ldconfig && \
    cd .. && \
    rm -rf ./ta-lib*

# clean up
RUN sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

RUN pip install freqtrade[all]

RUN pip install moviepy

# add $HOME/.local/bin to PATH
ENV PATH="/home/jupyter/.local/bin:${PATH}"
