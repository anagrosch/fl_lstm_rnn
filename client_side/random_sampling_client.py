import os
import time
import pickle
import argparse
from socket import *

SERVER_IP = "127.0.0.1" #change to server's public ip address
CLIENT_IP = "127.0.0.1" #change to client's public ip address
PARAM_PATH = os.path.join(os.getcwd(), "outputs", "best_model_params.pkl")

class ClientSocket:
	"""
	Class for client TCP socket receiving aggregated weights from server.
	"""
	def __init__(self, connection, server_info, buffer_size=1024, recv_timeout=5):
		self.connection = connection
		self.server_info = server_info
		self.buffer_size = buffer_size
		self.recv_timeout = recv_timeout

	def run(self):
		"""
		Function to get aggregated weights and save to a tmp parameter file.
		"""
		f = open("tmp.pkl", 'wb')
		count = 0
		while True:
			self.start_time = time.time()
			received_data, status = self.recv_data()

			if received_data != None:
				f.write(received_data)
				print("Received chunk {num} from server.".format(num=count))
				count += 1

			if status == 0:
				self.confirm_data()

				f.close()
				print('\nAggregated results saved to best_model_params.pkl')

				self.connection.close()
				print("Connection closed with {server} due to inactivity or error.".format(server=self.server_info))
				break


	def recv_data(self):
		"""
		Function to call recv() until all data received from server.
		"""
		received_data = b''
		while True:
			try:
				self.connection.settimeout(self.recv_timeout)
				data = self.connection.recv(self.buffer_size)
				received_data += data

				if data == b'':
					received_data = b''
				
					if (time.time() - self.start_time) > self.recv_timeout:
						return None, 0 #connection inactive

				elif str(data)[-2] == '.':
					if len(received_data) > 0:
						try:
							return received_data, 1

						except BaseException as e:
							print("Error decoding client data: {msg}.\n".format(msg=e))
							return None, 0

				else: self.start_time = time.time() #reset timeout counter
	
			except BaseException as e:
				print("Error receiving data: {msg}.\n".format(msg=e))
				return None, 0


	def confirm_data(self):
		"""
		Function to send confirmation of received data to server.
		"""
		msg = "Data received."
		msg = pickle.dumps(msg)
		self.connection.sendall(msg)
		print('Sent data confirmation to server.')


def receive_data_from(soc):
	"""
	Function to receive data from socket.
	"""
	received_data = b''
	while str(received_data)[-2] != '.':
		data = soc.recv(8)
		received_data += data

	received_data = pickle.loads(received_data)
	return received_data


def init_comm(server_port=10800):
	"""
	Function to initialize communication with server.
	"""
	print('\n-----------------------------------------')
	print('Initializing server-client communication.')
	print('-----------------------------------------\n')

	clientSocket = socket(AF_INET, SOCK_STREAM) #IPv4, TCP
	clientSocket.connect((SERVER_IP, server_port))
	print('Connected to server')

	# get confirmation from server
	status = receive_data_from(clientSocket)
	print("Received status from server: {data}\n".format(data=status))

	clientSocket.close()
	print('Client socket closed')


def client_send(server_port=10800):
	"""
	Function to send trained model and receive updated weights with
	TCP socket.
	"""
	print('\n-------------------------------------------------')
	print('Sending local model parameters to central server.')
	print('-------------------------------------------------\n')

	clientSocket = socket(AF_INET, SOCK_STREAM) #IPv4, TCP
	clientSocket.connect((SERVER_IP, server_port))
	print('Connected to server')

	# get selection status from server
	status = receive_data_from(clientSocket)
	print("Selection status from server: {status}".format(status=status))

	if status == "Selected for sampling":
		# send weights from each parameter file
		send_chunks(clientSocket, PARAM_PATH)
		print('Client sent model parameters to the server.')

		# get confirmation from server
		status = receive_data_from(clientSocket)
		print("Received status from server: {data}\n".format(data=status))

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


def client_get(client_port=12000):
	"""
	Function to get aggregated weights from server.
	"""
	print('\n---------------------------------------------------')
	print('Waiting for aggregated results from central server.')
	print('---------------------------------------------------\n')

	soc = socket(AF_INET, SOCK_STREAM)
	soc.bind((CLIENT_IP, client_port))
	print('Socket created')

	soc.listen(1)
	print('Listening for connection...')

	try:
		while True:
			# wait for connection with server
			connection, address = soc.accept()
			if address[0] == SERVER_IP:
				break
			else:
				print("Rejected connection from server: {addr}".format(addr=address))

		print("Accepted connection from server: {addr}\n".format(addr=address))

		client = ClientSocket(connection=connection,
				      server_info=address,
				      buffer_size=1024,
				      recv_timeout=5)
		client.run()

	except:
		soc.close()
		print('Socket closed.')

	update_params()
	print("Local parameters updated.")


def update_params():
	"""
	Function to average aggregated params from server with local params.
	"""
	with open("tmp.pkl", 'rb') as f:
		# get aggregated params
		tmp_dict = pickle.load(f)

	with open(PARAM_PATH, 'rb') as f:
		# get local params
		dict = pickle.load(f)

	for weight in dict.keys():
		dict[weight] = (dict[weight] + tmp_dict[weight])/2

	with open(PARAM_PATH, 'wb') as f:
		pickle.dump(dict, f)

	os.remove("tmp.pkl")


if __name__ == "__main__":
	"""
	Run basic client-server parameter aggregation.
	"""
	parser = argparse.ArgumentParser(prog='CLIENT SOCKET', usage='%(prog)s [options]')
	parser.add_argument('-i', '--init', action='store_true', help='initialize server-client connection')
	parser.add_argument('-s', '--send', action='store_true', help='send local model parameters to server')
	parser.add_argument('-g', '--get', action='store_true', help='start socket to get data from server')
	args = parser.parse_args()

	if not(args.init or args.send or args.get):
		print('Error: No action chosen.')
		print('<python3 random_sampling_client.py --help> for help')

	if (args.init and (args.send or args.get)):
		print('Error: Cannot run -i/--init with other flags.')
		print('Run <python3 random_sampling_client.py --init> first')
		raise SystemExit(1)

	# initialize server-client connection
	if args.init:
		init_comm()

	# send trained model parameters to central server
	if args.send:
		client_send()

	# get aggregated results from server and update model parameters
	if args.get:
		client_get()
