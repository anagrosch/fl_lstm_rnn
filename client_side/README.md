# Client Side of Federated Learning

Client sends trained model weights to a central server and
waits for aggregated results.

## Project Files

This folder has the following file:

- `client_socket.py`: Client app to send client's trained model
parameters to the central server. Waits for aggregated results
from the central server. Can be used by each client.

## Usage

### Send parameters to central server

Client gets model weights from local file `/outputs/best_model_params.pkl`
as a Python dictionary. Dictionary is sent to the central server in chunks.

Must change `client_ip` to IP address or hostname of client.

To execute, run command:
```
python3 client_socket.py --send
```
or
```
python3 client_socket.py -s
```

### Get aggregated results

Waits to receive aggregated results from the central server. Saves results
to `outputs/best_model_params.pkl`.

Will only accept connections with the server's IP defined by `server_ip`.

This file must be in the same location as the client's model.

To execute, run command:
```
python3 client_socket.py --get
```
or
```
python3 client_socket.py -g
```
