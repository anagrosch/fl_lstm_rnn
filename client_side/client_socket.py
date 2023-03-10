import os
import time
import pickle
import argparse
from socket import *

param_path = os.path.join(os.getcwd(), "outputs", "best_model_params.pkl")

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
		Function to get aggregated weights and update parameter file.
		"""
		f = open(param_path, 'wb')
		count = 0
		while True:
			self.start_time = time.time()
			received_data, status = self.recv_data()

			if received_data != None:
				f.write(received_data)
				print("Received chunk {num} from server.".format(num=count))
				count += 1

			if status == 0:
				f.close()
				print('Aggregated results saved to best_model_params.pkl')

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


def client_send(server_port=10800, server_ip="192.168.1.60"):
	"""
	Function to send trained model and receive updated weights with
	TCP socket.
	"""
	print('-------------------------------------------------')
	print('Sending local model parameters to central server.')
	print('-------------------------------------------------')

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


def client_get(server_ip="192.168.1.60"):
	"""
	Function to get aggregated weights from server.
	"""
	print('')
	print('---------------------------------------------------')
	print('Waiting for aggregated results from central server.')
	print('---------------------------------------------------')

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
			if address[0] == server_ip:
				print("Accepted connection from server: {addr}".format(addr=address))

				client = ClientSocket(connection=connection,
						      server_info=address,
						      buffer_size=1024,
						      recv_timeout=10)
				client.run()

			else:
				print("Rejected connection from server: {addr}".format(addr=address))

		except:
			soc.close()
			print('Socket closed.')
			break


# main function
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--send', action='store_true', help='send local model parameters to server')
parser.add_argument('-g', '--get', action='store_true', help='start socket to get data from server')
args = parser.parse_args()

# send trained model parameters to central server
if args.send:
	client_send()
	time.sleep(1)

# get aggregated results from server and update model parameters
if args.get:
	client_get()

if not(args.send or args.get):
	print('Error: No action chosen')
	print('<python3 client_socket.py --help> for help')
