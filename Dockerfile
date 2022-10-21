# Base image
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Base packages
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Gothenburg
RUN \
  apt-get update -qy && \
  apt-get upgrade -qy && \
  apt-get install -qy \
    git \
    graphviz \
    pandoc \
    python3-pip \
    zip

RUN \
  pip3 install --upgrade \
    pip \
  && \
  pip3 install --upgrade \
    coverage \
    flake8 \
    nbmake \
    pytest \
    setuptools_scm \
    twine \
    xdoctest

# Packages needed for calorine (compare setup.py)
RUN \
  pip3 install \
    ase \
    matplotlib \
    numpy \
    pandas \
    pybind11

# Packages for building documentation
RUN \
  pip3 install --upgrade \
    sphinx_autodoc_typehints \
    sphinx-rtd-theme \
    sphinx_sitemap \
    sphinxcontrib-bibtex \
    cloud_sptheme \
    nbsphinx \
  && \
  pip3 install --upgrade \
    jinja2==3.0.3

CMD /bin/bash
