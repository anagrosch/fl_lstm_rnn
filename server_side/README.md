# Central Server Side of Federated Learning

Central server collects model weights from multiple clients,
aggregates the data, then redistributes the weights to the clients.

## Project Files

This folder has the following files:

- `basic_server.py`: Socket app to create `pkl` files with model
parameters received from each client. Sends aggregated results back
to each client. Executes `aggregate.py` to aggregate the collected
client models.

- `random_sampling_server.py`: Socket app with random sub-sampling of
clients. Uses `basic_server.py` as base foundation.

- `aggregate.py`: Aggregates each client's model parameters and saves
the final results to a local file. Saves collection, aggregation, and
redistribution times to a csv file.

## Usage

### Basic Socket Program

#### Initialization

Opens server socket and waits for connections from clients to be used for
aggregation. Saves each client's IP address as a Python dictionary. Each
key is assigned 0.

The Python dictionary is saved to `/outputs/client_info.pkl`.

Sends the model parameters currently saved to `/outputs/final_weights.pkl` to
each client.

Must run initialization before collecting client models.
Only need to run when adding new clients to the approved client dictionary.

Cannot run with another flag set.

This functionality is currently commented out.

To execute, run command:
```
python3 basic_server.py --init
```
or
```
python3 basic_server.py -i
```

#### Get Client Parameters

When a connection is established between the central server and client(s),
the server creates a file named `<client_port>_model.pkl` for each client.
If the file already exists, the `client_port` is incremented by 1 until the
file name is available.

The trained model received by each client is saved to the client's
respective file.

Client files are saved to `/client_models/`.

The IP addresses of each client are saved to `/outputs/client_info.pkl`
as a Python dictionary.

The server sends a confirmation of received data to the clients, including the
number of data chunks received.

Supports multithreading.

User must close socket when no more clients are to connect with `Ctrl+C`.

To execute, run command:
```
python3 basic_server.py --get
```
or
```
python3 basic_server.py -g
```

#### Redistribute Aggregated Results to Clients

Server gets aggregated results from local file `/outputs/final_weights.pkl`
and sends the data to each client address provided in local file
`client_info.pkl`.

Must change `SERVER_IP` to IP address or hostname of server.

Must start client process first to receive the data.

To execute, run command:
```
python3 basic_server.py --send
```
or
```
python3 basic_server.py -s
```

### Random Sub-Sampling Socket Program

#### Initialize

Opens server socket and waits for connections from clients to be used for
aggregation. Saves each client's IP address as a Python dictionary. Each
key is assigned 0.

The Python dictionary is saved to `/outputs/client_info.pkl`.

Sends the model parameters currently saved to `/outputs/final_weights.pkl` to
each client. (This functionality is currently commented out)

Must run initialization before collecting client models.
Only need to run when adding new clients to the approved client dictionary.

Cannot run with another flag set.

To execute, run command:
```
python3 random_sampling_socket.py --init
```
or
```
python3 random_sampling_socket.py -i
```

#### Get Client Parameters

Server gets approved client addresses from Python dictionary saved in
`/outputs/client_info.pkl`. Approximately 2/3 of the clients are chosen
for the sample, which are then assigned a value of 1.

A value of 1 means the client's model will be used in the next round of
aggregation. A value of 0 means the client is not part of the next sample.

When a connection is established between the central server and client(s),
the server checks if the client's IP address is in the dictionary saved to
file `/outputs/client_info.pkl`.
If it is not, the connection is refused.

If the value saved to the client's IP address is a 0, the server tells the
client it is not part of the round's sample and closes the connection.

If the value saved to the client's IP address is a 1, the server continues
the steps described in Get Client Parameters in
[Basic Socket Program](#basic-socket-program).

The values for each client's IP address determined earlier is saved to file
`/outputs/client_info/pkl`.

To execute, run command:
```
python3 random_sampling_server.py --get
```
or
```
python3 random_sampling_server.py -g
```

#### Redistribute Aggregated Results to Clients

Server gets client IP addresses from dictionary in file `/outputs/client_info.pkl`.

Server sends aggregated results from file `/outputs/final_weights.pkl` to each
client whose dictionary value is 1.

Must change `SERVER_IP` to IP address or hostname of server.

Must start client process first.

To execute, run command:
```
python3 random_sampling_server.py --send
```
or
```
python3 random_sampling_server.py -s
```

### Aggregation Program

Reads each client's parameters from client files located in `/client_models/`.
Aggregates new client parameters with equal weights.

If there are weights saved in file `/outputs/final_weights.pkl` from a
previous round of aggregation, the weights are aggregated with the
new client weights.

Final aggregated results saved to `/outputs/final_weights.pkl`.

The `/client_models/` directory with the client weights is deleted
after aggregation.

To execute with the basic server, run command:
```
python3 basic_server.py --aggr
```
or
```
python3 basic_server.py -a
```

To execute with random sub-sampling, run command:
```
python3 random_sampling_server.py --aggr
```
or
```
python3 random_sampling_server.py -a
```

