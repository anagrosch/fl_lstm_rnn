import os
import pickle
from socket import *
from lstm_model import Model
from utils import get_params

param_path = os.path.join(os.getcwd(), "outputs", "best_model_params.pkl")

def client_send(server_port=10800, server_ip="192.168.1.60"):
	"""
	Function to send trained model and receive updated weights with
	TCP socket.
	"""
	data = add_addr_to_dict()

	clientSocket = socket(AF_INET, SOCK_STREAM) #IPv4, TCP
	clientSocket.connect((server_ip, server_port))
	print('Connected to server')

	# Send model data to server
	data = pickle.dumps(data)
	clientSocket.sendall(data)
	print('Client sent model parameters to the server')

	received_data = b''
	while str(received_data)[-2] != '.':
		data = clientSocket.recv(8)
		received_data += data

	received_data = pickle.loads(received_data)
	print("Received status from server: {data}".format(data=received_data))

	clientSocket.close()
	print('Client socket closed')


def add_addr_to_dict():
	"""
	Function to get weight dictionary from file and add client's address
	to the dictionary. Returns dictionary.
	"""
	client_port = 12000
	client_ip = ""

	with open(param_path, 'rb') as f:
		dict = pickle.load(f)

	dict[client_ip] = client_port
	return dict


def client_get():
	"""
	Function to get aggregated weights from server.
	"""
	server_ip = "192.168.1.60"
	server_port = 10800

	client_ip = "192.168.1.60" #change to machine's ip
	client_port = 12000

	soc = socket(AF_INET, SOCK_STREAM)
	soc.bind((client_ip, client_port))
	print('Socket created')

	soc.listen(1)
	print('Listening for connection...')

	while True:
		try:
			connection, address = soc.accept()
			if address = (server_ip, server_port):
				print("Accepted connection from server: {addr}".format(addr=address))
			
				received_data = b''
				while str(received_data0[-2] != '.':
					data = soc.recv(8)
					received_data += data

				received_data = pickle.loads(received_data)
				print('Received aggregated results from server.')

				with open(param_path, 'wb') as f:
					pickle.dump(received_data, f)
				print('Aggregated results saved to best_model_params.pkl')
	
			else:
				print("Rejected connection from server: {addr}".format(addr=address))

		except:
			soc.close()
			print('Socket closed.')
			break


# send trained model parameters to central server
client_send()

# get aggregated results from server and update model parameters
client_get()
