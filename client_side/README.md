# Client Side of Federated Learning

Client sends trained model weights to a central server and
waits for aggregated results.

## Project Files

This folder has the following files:

- `basic_client.py`: Client app to send client's trained model
parameters to the central server. Waits for aggregated results
from the central server. Can be used by each client.

- `random_sampling_client.py`: Client app for random sub-sampling of
clients. Uses `basic_client.py` as base foundation.

## Usage

### Initialization

Client connects to server and receives server's current model parameters.
Parameters saved to `/outputs/best_model_params.pkl`.
This functionality is currently commented out for
`random_sampling_client.py`.

Run before any client-server interaction occurs.
If done before, no need to run again, unless server resets.

Cannot run with another flag set.

This functionality is currently commented out for `basic_client.py`.

To execute with the basic client, run command:
```
python3 basic_client.py -init
```
or
```
python3 basic_client.py -i
```

To execute with random sub-sampling, run command:
```
python3 random_sampling_client.py --init
```
or
```
python3 random_sampling_client.py -i
```

### Random Sub-Sampling

Only applicable for `random_sampling_client.py`.

Client waits for sampling selection from server.

If client is not selected, connection is closed.

If client is selected, client continues with steps described in
[Send Parameters To Central Server](#send-parameters-to-central-server)

### Send Parameters To Central Server

Client gets model weights from local file `/outputs/best_model_params.pkl`
as a Python dictionary. Dictionary is sent to the central server in chunks.

Must change `client_ip` to IP address or hostname of client.

To execute with the basic client, run command:
```
python3 basic_client.py --send
```
or
```
python3 basic_client.py -s
```

To execute with random sub-sampling, run command:
```
python3 random_sampling_client.py --send
```
or
```
python3 random_sampling_client.py -s
```

### Get Aggregated Results

Waits to receive aggregated results from the central server. Saves results
to `outputs/best_model_params.pkl`.

Will only accept connections with the server's IP defined by `server_ip`.

This file must be in the same location as the client's model.

To execute with the basic client, run command:
```
python3 basic_client.py --get
```
or
```
python3 basic_client.py -g
```

To execute with random sub-sampling, run command:
```
python3 random_sampling_client.py --get
```
or
```
python3 random_sampling_client.py -g
```

