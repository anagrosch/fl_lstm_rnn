# Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)

A simple lstm rnn model that predicts the rest of the sentence given a number of words.

## Project Files

- `requirements.txt`: Project dependencies

- `lstm_model.py`: PyTorch model for an lstm rnn

- `dataset.py`: Dataset class

- `train_lstm.py`: Trains model and saves model checkpoints for the final trained model and the
trained model point with the smallest validation loss

- `utils.py`: Class to save and load the model checkpoints, create training vs. validation loss
plots, and save the parameters of the best model checkpoint.

## Usage

The model reads multiple sentences from an input csv file and divides the dataset into a training
and validation dataset.

During training, two stages of the model are saved for future training and/or prediction. After training
is completed, the final model is saved in '/outputs/best_model.pth'. During training, the validation loss
is monitored so that the epoch with the lowest validation loss is saved in `/outputs/best_model.pth`.

The training and validation loss at each epoch are plot and saved to `/outputs/`.

The model parameters are saved as a Python dictionary in `/outputs/best_model_params.pkl`.

## Setup

### Install Pytorch

PyPi does not have an official PyTorch package for ARM architectures (Raspberry Pi), but there are
a few different ways to download and install PyTorch. This guides you to install PyTorch with a
pre-compiled wheel file.

To know which wheel file to download, you can check the pi's compatible version of processor
architecture (i.e. armv71) using the command `uname -a`.

You must also ensure the gcc version matches the processor architecture with the command `gcc -v`.

Buster OS and Bullseye OS were used in this project, so the steps to install
PyTorch on both OS types are provided below.

#### Buster OS

1. Install PyTorch dependencies.
```
sudo apt install libopenblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
```
2. Download the [wheel file](https://drive.google.com/file/d/1D3A5YSWiY-EnRWzWbzSqvj4YdY90wuXq/view) onto
your pi.

3. Go to the directory with the wheel file.

4. Install PyTorch.
```
pip3 install <torch_file_name.whl>
```

#### Bullseye OS

1. Install PyTorch dependencies.
```
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
```
```
sudo -H pip3 install setuptools==58.3.0
sudo -H pip3 Cython
```
2. Download the [wheel file](https://drive.google.com/uc?id=1uLkZzUdx3LiJC-Sy_ofTACfHgFprumSg) onto
your pi.

3. Go to the directory with the wheel file.

4. Install PyTorch.
```
sudo -H pip3 install <torch_file_name.whl>
```

To check if installation was successful, run:
```
python3
import torch
```

### Install Project Dependencies

Install necessary dependencies for the LSTM RNN model with the following command:
```
pip3 install -r requirements.txt
```

## Resources

- [Saving and Loading the Best Model in PyTorch](https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/)

- [PyTorch LSTM: Text Generation Tutorial](https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html)

- [Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html)

- [Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

- [Difference Between PyTorch and PySyft](https://analyticsindiamag.com/difference-between-pytorch-and-pysyft/)

- [Install PyTorch on a Raspberry Pi 4](https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html)

- [A Step by Step guide to installing PyTorch in Raspberry Pi](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531)
