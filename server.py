from socket import *
import pickle
import time
import os
import threading

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
		while True:
			self.start_time = time.time()
			received_data, status = self.recv_data()

			if received_data != None:
				# create local file
				port_num = client_info[1]
				filename = str(port_num) + "_model.pth"

				# rename file if already exists
				while(os.path.exists(os.getcwd()+filename)):
					port_num += 1
					filename = str(port_num) + "_model.pth"

				# write received data to file
				with open(filename, 'w') as f:
					f.writelines(received_data)

				msg = "Server received model data"
				msg = pickle.dumps(msg)
				connection.sendall(msg)
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
				data = connection.recv(self.buffer_size)
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


serverIP = "192.168.1.60"
serverPort = 10800

serverSocket = socket(AF_INET,SOCK_STREAM)
serverSocket.bind((serverIP,serverPort))
print('Socket created')

serverSocket.listen(1)
print('Listening for connection...')

while True:
	try:
		connection, client_info = serverSocket.accept()
		print("New connection from client: {client_info}.".format(client_info=client_info))
		socket_thread = SocketThread(connection=connection, client_info=client_info,
					     buffer_size=1024, recv_timeout=10)
		socket_thread.start()

	except:
		serverSocket.close()
		print('Socket closed. Server received no connections')
		break
