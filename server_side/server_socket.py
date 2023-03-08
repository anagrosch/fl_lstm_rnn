import os, time
import pickle
import threading
from socket import *
from os.path import join, exists
from aggregate import get_dict

model_dir = join(os.getcwd(), "client_models")

class SocketThread(threading.Thread):
	"""
	Class for multithreading TCP socket
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
		while True:
			self.start_time = time.time()
			received_data, status = self.recv_data()

			if received_data != None:
				# create local file
				port_num = self.client_info[1]
				file_path = join(model_dir,str(port_num)+"_model.pkl")

				# rename file if already exists
				while(exists(file_path)):
					port_num += 1
					file_path = join(model_dir,str(port_num)+"_model.pkl")

				# write received data to file
				with open(file_path, 'wb') as f:
					pickle.dump(received_data, f)

				msg = "Server received model data"
				msg = pickle.dumps(msg)
				self.connection.sendall(msg)
				print('Server sent model data confirmation to client')

			if status == 0:
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
				data = self.connection.recv(self.buffer_size)
				received_data += data

				if data == b'':
					received_data = b''

					if (time.time() - self.start_time) > self.recv_timeout:
						return None, 0 #connection inactive
			
				elif str(data)[-2] == '.':
					print('All data received')

					if len(received_data) > 0:
						try:
							received_data = pickle.loads(received_data)
							return received_data, 1

						except BaseException as e:
							print("Error decoding client data: {msg}.\n".format(msg=e))
							return None, 0
	
				else: self.start_time = time.time() #reset timeout counter

			except BaseException as e:
				print("Error receiving data: {msg}.\n".format(msg=e))
				return None, 0


def server_get():
	"""
	Function to get client weights with a TCP socket
	"""
	server_ip = "192.168.1.60"
	server_port = 10800

	serverSocket = socket(AF_INET,SOCK_STREAM)
	serverSocket.bind((server_ip,server_port))
	print('Socket created')

	serverSocket.listen(1)
	print('Listening for connection...')

	while True:
		try:
			connection, client_info = serverSocket.accept()
			print("New connection from client: {client_info}".format(client_info=client_info))

			# create directory to save models to
			if not exists(model_dir):
				os.mkdir(model_dir)
				print("Client model directory created.")

			socket_thread = SocketThread(connection=connection,
						     client_info=client_info,
						     buffer_size=1024,
						     recv_timeout=10)
			socket_thread.start()

		except:
			serverSocket.close()
			print('Socket closed. Server received no connections')
			break


def server_send(addr_dict, weight_path):
	"""
	Function to send aggregated weights to clients.
	"""
	# iterate through each client address
	for ip, port in addr_dict.items():
		socket = socket(AF_INET, SOCK_STREAM)
		socket.connect((ip, port))
		print("Connected to client: ({ip}, {port})".format(ip=ip, port=port))

		# send weights to client
		data = get_dict(join(os.getcwd(), "final_weights.pkl"))
		data = pickle.dumps(data)
		socket.sendall(data)
		print("Server send weights to client: ({ip}, {port})".format(ip=ip, port=port))

		# get client confirmation
		received_data = b''
		while str(received_data)[-2] != '.':
			data = socket.recv(8)
			received_data += data

		received_data = pickle.loads(received_data)
		print("Received status from client: {data}".format(data=received_data))

		socket.close()
		print("Socket closed with client: ({ip}, {port})".format(ip=ip, port=port))

	print('Aggregated weights sent to all clients')


if __name__ == "__main__":
	server_socket()
