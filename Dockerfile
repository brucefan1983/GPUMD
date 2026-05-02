# Base image
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

# Base packages
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Gothenburg

RUN apt-get update -qy && \
    apt-get install -y --no-install-recommends \
      git \
      graphviz \
      pandoc \
      python3-pip \
      zip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
      coverage \
      flake8 \
      nbmake \
      pytest \
      setuptools_scm \
      twine \
      xdoctest \
      ase \
      matplotlib \
      numpy \
      pandas \
      pybind11 \
      sphinx_autodoc_typehints \
      sphinx-rtd-theme \
      sphinx_sitemap \
      sphinxcontrib-bibtex \
      nbsphinx

CMD ["/bin/bash"]
