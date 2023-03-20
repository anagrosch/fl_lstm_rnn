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

PyPi does not have an official PyTorch package for ARM architectures (Raspberry Pi). There are
a few different ways to download and install PyTorch. We're going to focus on installing PyTorch
with a pre-compiled wheel file. Here is an
[arm71 compatible wheel file](https://drive.google.com/file/d/1D3A5YSWiY-EnRWzWbzSqvj4YdY90wuXq/view).

To know which wheel file to download, you can check the pi's compatible version of processor
architecture (i.e. armv71).
```
uname -a
```
Install PyTorch dependencies.
```
sudo apt install libopenblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
```
Download the [wheel file](https://drive.google.com/file/d/1D3A5YSWiY-EnRWzWbzSqvj4YdY90wuXq/view) onto
your pi.

Go to the directory with the wheel file.

Install PyTorch.
```
pip3 install <torch_file_name.whl>
```

To check if installation was successful, run:
```
python3
import torch
```

### Install PySyft

Install PyAV version 8.0.0 (or newer).
```
sudo apt install -y libavdevice-dev
pip3 install av >=8.0.0
```
Install additional PySyft dependencies.
```
sudo apt install libavfilter-dev libopus-dev libvpx-dev pkg-config
pip3 install toml
pip3 install cffi==1.15.0
```
Go to the [rustup website](rustup.rs) and follow the directions for installation. Close the terminal
after installing.
```
pip install --upgrade pip
python3 -m pip install sycret
pip3 install websocket-client
```

## Resources

- [Saving and Loading the Best Model in PyTorch](https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/)

- [PyTorch LSTM: Text Generation Tutorial](https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html)

- [Difference Between PyTorch and PySyft](https://analyticsindiamag.com/difference-between-pytorch-and-pysyft/)

- [A Step by Step guide to installing PyTorch in Raspberry Pi](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531)

- [A Step by Step guide to installing PySyft in Raspberry Pi](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pysyft-in-raspberry-pi-d8d10c440c37)
