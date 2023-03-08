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

When a connection is established between the central server and client(s),
the server creates a file named `<client_port>_model.pkl` for each client.
If the file already exists, the `client_port` is incremented by 1.

The trained model received by each client is saved to the client's
respective file.

Client files are saved to `/client_models/`.

Gets aggregated results from local file `final_weights.pkl` and sends the
data to each client address provided in a dictionary.

Supports multithreading.

Must change `server_ip` to IP address or hostname of server.

### Aggregation Program

Calls `server_socket.py` to get model parameters from each client.

Gets each client's parameters from `.pkl` files in the `client_models`
directory. Aggregates the parameters and saves the final results to
`final_weights.pkl`.

Gets each client's IP address and port number from each client's
parameter file. Calls `server_socket.py` to send aggregated results
to each client.

Deletes client files after aggregation.

To execute, run command:
```
python3 aggregate.py
```
