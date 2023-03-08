# Client Side of Federated Learning

Client sends trained model weights to a central server and
waits for aggregated results.

## Project Files

This folder has the following file:

- `client_socket.py`: Client app to send client's trained model
parameters to the central server. Waits for aggregated results
from the central server. Can be used by each client.

## Usage

Client gets model weights from local file `/outputs/best_model_params.pkl`
as a Python dictionary. Adds client's IP address and port number to the
dictionary. Sends dictionary to the central server.

Waits to receive aggregated results from the central server. Saves results
to `outputs/best_model_params.pkl`.

This file must be in the same location as the client's model.

To execute, run command:
```
python3 client_socket.py
```
