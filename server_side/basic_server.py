import os
import time
import argparse
import pickle
import threading
from socket import *
from os.path import join, exists
from aggregate import aggr_params, save_times

SERVER_IP = "127.0.0.1" #change to server's public ip address
MODEL_DIR = join(os.getcwd(), "client_models")

class SocketThread(threading.Thread):
	"""
	Class for multithreading TCP socket.
	"""
	def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
		threading.Thread.__init__(self)
		self.connection = connection
		self.client_info = client_info
		self.buffer_size = buffer_size
		self.recv_timeout = recv_timeout

	def run(self):
		"""
		Function to get client weights and create local files for each client
		"""
		# create local file
		port_num = self.client_info[1]
		file_path = join(MODEL_DIR,str(port_num)+"_model.pkl")

		# rename file if already exists
		while(exists(file_path)):
			port_num += 1
			file_path = join(MODEL_DIR,str(port_num)+"_model.pkl")

		f = open(file_path, 'wb')
		self.param_count = 0
		while True:
			self.start_time = time.time()
			received_data, status = self.recv_data()

			if received_data != None:
				# write received data to file
				f.write(received_data)

				print("Received chunk {num} from client: {client}".format(num=self.param_count,client=self.client_info))
				self.param_count += 1

			if status == 0:
				self.confirm_data()

				f.close()
				self.connection.close()
				print("Connection closed with {client} due to inactivity or error.".format(client=self.client_info))
				break


	def recv_data(self):
		"""
		Function to call recv() until all data received from client.
		outputs:	received_data=data from client
				1=keep connection open, 0=close connection
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
				print("Error receiving data from {client}: {msg}.\n".format(client=self.client_info,msg=e))
				return None, 0


	def confirm_data(self):
		"""
		Function to send confirmation of received data to client.
		"""
		msg = "Server received {num} chunks.".format(num=self.param_count)
		msg = pickle.dumps(msg)
		self.connection.sendall(msg)
		print("Sent data confirmation to client: {client}".format(client=self.client_info))


def server_get(server_port=10800):
	"""
	Function to get client weights with a TCP socket
	"""
	print('\n------------------------------------------')
	print('Waiting for model parameters from clients.')
	print('------------------------------------------\n')

	info_path = join(os.getcwd(),"outputs","client_info.pkl")

	# create directory to save models to
	if not exists(MODEL_DIR):
		os.mkdir(MODEL_DIR)
		print("Client model directory created.")

	# create directory for outputs
	if not exists("outputs"):
		os.mkdir("outputs")
		print("Outputs directory created.")

	# create dictionary to hold client info
	addr_dict = {}

	serverSocket = socket(AF_INET,SOCK_STREAM)
	serverSocket.bind((SERVER_IP,server_port))
	print('Socket created')

	serverSocket.listen(1)
	print('Listening for connection...')

	while True:
		try:
			connection, client_info = serverSocket.accept()
			print("New connection from client: {client_info}".format(client_info=client_info))

			addr_dict[client_info[0]] = client_info[1]

			socket_thread = SocketThread(connection=connection,
						     client_info=client_info,
						     buffer_size=1024,
						     recv_timeout=5)
			socket_thread.start()

		except:
			serverSocket.close()
			print('Socket closed. Server received no connections.\n')
			break

	# write client info dictionary to file
	with open(info_path, 'wb') as f:
		pickle.dump(addr_dict, f)
	print('Client addresses saved to file: /outputs/client_info.pkl')


def server_send(client_port=12000):
	"""
	Function to send aggregated weights to clients.
	"""
	print('\n--------------------------------------')
	print('Sending aggregated results to clients.')
	print('--------------------------------------\n')

	buffer_size = 1024
	info_path = join(os.getcwd(),"outputs","client_info.pkl")
	weight_path = join(os.getcwd(),"outputs","final_weights.pkl")

	# get client addresses
	with open(info_path, 'rb') as f:
		addr_dict = pickle.load(f)

	# iterate through each client address
	soc = socket(AF_INET, SOCK_STREAM)
	for ip in addr_dict.keys():
		soc.connect((ip, client_port))
		print("Connected to client: ({ip}, {port})".format(ip=ip, port=client_port))

		# send weights to client in chunks
		send_chunks(soc, weight_path)
		print("Server send weights to client: ({ip}, {port})".format(ip=ip,port=client_port))

		# get confirmation from clients
		received_data = b''
		while str(received_data)[-2] != '.':
			data = soc.recv(8)
			received_data += data

		received_data = pickle.loads(received_data)
		print("Received status from ({ip}, {port}): {data}\n".format(ip=ip,port=client_port,data=received_data))

		soc.close()
		print("Socket closed with client: ({ip}, {port})".format(ip=ip, port=client_port))

	print('Aggregated weights sent to all clients')


def send_chunks(soc, path):
	"""
	Function to split pickle file into chunks and send to clients.
	"""
	buffer_size = 1024
	f = open(path, 'rb')
	while True:
		chunk = f.read(buffer_size)
		if not chunk:
			f.close()
			break
		soc.sendall(chunk)


# main function
parser = argparse.ArgumentParser(prog='BASIC SOCKET',
				 usage='%(prog)s [options]',
				 description='Basic server socket to get client models and send aggregated weights.')
parser.add_argument('-g', '--get', action='store_true', help='start socket to get data from clients')
parser.add_argument('-a', '--aggr', action='store_true', help='aggregate client weights')
parser.add_argument('-s', '--send', action='store_true', help='send aggregated results to clients')
args = parser.parse_args()

if not(args.get or args.aggr or args.send):
	print('Error: No action chosen')
	print('<python3 basic_socket.py --help> for help')
	raise SystemExit(1)

# get trained models from clients
if args.get:
	start_time = time.perf_counter()
	server_get()
	get_time = time.perf_counter() - start_time

# aggregate weights
if args.aggr:
	start_time = time.perf_counter()
	aggr_params()
	aggr_time = time.perf_counter() - start_time

# send aggregated weights
if args.send:
	start_time = time.perf_counter()
	server_send()
	send_time = time.perf_counter() - start_time

save_times(get_time, aggr_time, send_time)
