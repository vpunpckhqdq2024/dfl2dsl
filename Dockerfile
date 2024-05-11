from ubuntu:22.04

RUN mkdir dfl2dsl
RUN apt update && apt install -y curl build-essential unzip bubblewrap wget libcairo2-dev libzmq3-dev pkg-config
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rust.sh && chmod +x rust.sh && ./rust.sh -y
RUN bash -c "echo | sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)" && \
    opam init --disable-sandboxing -y
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh -b && \
    echo "export PATH=\"/root/miniconda3/condabin:$PATH\"" >> /root/.bashrc
COPY environment.yml environment.yml
RUN /root/miniconda3/condabin/conda install -n base conda-libmamba-solver && \
    /root/miniconda3/condabin/conda env create -f environment.yml --solver=libmamba
RUN /root/miniconda3/condabin/conda run -n dfl2dsl python3 -m pip install pregex==1.0.0 --ignore-requires-python
WORKDIR dfl2dsl
