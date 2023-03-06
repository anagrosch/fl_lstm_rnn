# Federated Learning Implementation of LSTM RNN

Send client models to central server with IPv4 TCP sockets.

## Project Files

This project has the following files:

- `server.py`: Central server app to create `pth` files with models
received from each client. Must be executed before all clients.

- `client.py`: Client app to send client's trained model to the central
server. Can be used by each client.

## Usage

### Server Side

When a connection is established between the central server and client(s), server
creates a file named `<client_port>_model.pth` for each client. If the file already
exists, the client_port is incremented by 1.

The trained model received by each client is saved to the client's respective file.

Supports multithreading.

Must change `serverIP` to IP address or hostname of server.

To execute, run command:
```
python3 server.py
```

### Client Side

Client sends its trained model to central server.

To execute, run command:
```
python3 client.py <model>.pth
```
where `<model>.pth` is the local file with the client's trained model information.
