import os
import pickle
from socket import *

param_path = os.path.join(os.getcwd(), "outputs", "best_model_params.pkl")

def client_send(server_port=10800, server_ip="192.168.1.60"):
	"""
	Function to send trained model and receive updated weights with
	TCP socket.
	"""
	client_ip = "192.168.1.60" #change with machine's public ip address
	client_port = 12000

	clientSocket = socket(AF_INET, SOCK_STREAM) #IPv4, TCP
	clientSocket.connect((server_ip, server_port))
	print('Connected to server')

	# send weights from each parameter file
	send_chunks(clientSocket, param_path)
	print('Client sent model parameters to the server.')

	clientSocket.close()
	print('Client socket closed')


def send_chunks(soc, path):
	"""
	Function to split pickle files into chunks and send to server socket.
	"""
	buffer_size = 1024
	f = open(path,'rb')
	while True:
		# send chunk to server
		chunk = f.read(buffer_size)
		if not chunk:
			f.close()
			break
		soc.sendall(chunk)


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
			if address is (server_ip, server_port):
				print("Accepted connection from server: {addr}".format(addr=address))
			
				received_data = b''
				while str(received_data)[-2] != '.':
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
#client_get()
