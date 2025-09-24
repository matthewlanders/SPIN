# SPIN: Improving and Accelerating Offline RL in Large Discrete Action Spaces with Structured Policy Initialization

This repository contains the official SPIN implementation.
## How to run the code

### Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### Install environments and download datasets

Installation instructions for installing the DeepMind Control suite, installing the Maze environment and downloading the datasets can be found here.

### Run ASM Pre-Training
```
python ASM.py
```

### Run training/evaluation
```
python IQL.py
```