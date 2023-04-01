# QuakeSaver Neural Network Models

[![Export TorchScript models](https://github.com/QuakeSaver/torch-models/actions/workflows/export-models.yml/badge.svg)](https://github.com/QuakeSaver/torch-models/actions/workflows/export-models.yml)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

This repository assembles the neural network models for QuakeSaver seismic sensors.

The [pytorch](https://pytorch.org/) models are assembled in [TorchScript](https://pytorch.org/docs/stable/jit.html), which are executed on-the-edge on the QuakeSaver sensor's firmware.

## Available Models

We are actively working on bringing neural networks for seismology and building health monitoring to the edge.

### PhaseNet

PhaseNet phase picker for P- and S- wave phase picks trained by [SeisBench](https://github.com/seisbench/seisbench) on various large data sets:

* ethz
* geofon
* instance
* iquique
* lendb
* neic
* original
* scedc
* stead

For more information see

> Weiqiang Zhu, Gregory C Beroza, PhaseNet: a deep-neural-network-based seismic arrival-time picking method, Geophysical Journal International, Volume 216, Issue 1, January 2019, Pages 261–273, <https://doi.org/10.1093/gji/ggy423>

> Jack Woollam, Jannes Münchmeyer, Frederik Tilmann, Andreas Rietbrock, Dietrich Lange, Thomas Bornstein, Tobias Diehl, Carlo Giunchi, Florian Haslinger, Dario Jozinović, Alberto Michelini, Joachim Saul, Hugo Soto; SeisBench—A Toolbox for Machine Learning in Seismology. Seismological Research Letters 2022;; 93 (3): 1695–1709. doi: <https://doi.org/10.1785/0220210324>

## Development Installation

Local installation of the helper repositoy

```sh
pip3 install .
```

### Development

We utilize `pre-commit` for clean commits.

```sh
pre-commit install
```

## Distribution of Models

Distribution of the TorchScript models is carried out by GitHub actions. See the `.github/workflow` folder for details on continuous deployment. The sensors fetch models from the released artifacts.
