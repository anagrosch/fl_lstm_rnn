# Federated Learning Implementation of LSTM RNN

Clients send trained model parameters to central server.
The central server aggregates the data, then sends the
aggregated results to each client.
Communication done with IPv4 TCP sockets.

## Project Organization

The central server files are located in the `server_side` folder.

The client side files are located in the `client_side` folder.

Client must add the full model files to the same location as the
TCP socket file.

## Resources

- [Federated Learning Demo in Python][https://medium.com/cometheartbeat/federated-learning-demo-in-python-part-2-multiple-connections-using-threading-8d781d53e0c8]
