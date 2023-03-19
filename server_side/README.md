# Central Server Side of Federated Learning

Central server collects model weights from multiple clients,
aggregates the data, then redistributes the weights to the clients.

## Project Files

This folder has the following files:

- `server_socket.py`: Socket app to create `pkl` files with model
parameters received from each client. Sends aggregated results back
to each client.

- `aggregate.py`: Aggregates each client's model parameters and saves
the final results to a local file. Executes `server_socket.py` to get
each client's parameters and redistribute the aggregates results.

## Usage

### Socket Program

#### Get Client Parameters

When a connection is established between the central server and client(s),
the server creates a file named `<client_port>_model.pkl` for each client.
If the file already exists, the `client_port` is incremented by 1 until the
file name is available.

The trained model received by each client is saved to the client's
respective file.

Client files are saved to `/client_models/`.

The IP addresses of each client are saved to `/client_models/client_info.pkl`
as a Python dictionary.

The server sends a confirmation of received data to the clients, including the
number of data chunks received.

Supports multithreading.

User must close socket when no more clients are to connect with `Ctrl+C`.

To execute, run command:
```
python3 aggregate.py --get
```
or
```
python3 aggregate.py -g
```

#### Redistribute Aggregated Results to Clients

Server gets aggregated results from local file `final_weights.pkl` and sends the
data to each client address provided in local file `client_info.pkl`.

Must change `server_ip` to IP address or hostname of server.

Must start client process first to receive the data.

To execute, run command:
```
python3 aggregate.py --send
```
or
```
python3 aggregate.py -s
```

### Aggregation Program

Reads each client's parameters from client files located in `/client_models/`.
Aggregates new client parameters with equal weights.

If there are weights saved in local file `final_weights.pkl` from a
previous round of aggregation, the weights are aggregated with the
new client weights.

Final aggregated results saved to `final_weights.pkl`.

The `/client_models/` directory with the client weights is deleted
after aggregation.

To execute, run command:
```
python3 aggregate.py --aggr
```
or
```
python3 aggregate.py -a
```
