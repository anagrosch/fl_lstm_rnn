from socket import *
import sys
import pickle

serverIP = "192.168.1.60"
serverPort = 10800

filename = sys.argv[1]
with open(filename, 'r') as f:
	data = f.readlines()

clientSocket = socket(AF_INET, SOCK_STREAM) #IPv4, TCP
clientSocket.connect((serverIP, serverPort))
print('Connected to server')

# Send model data to server
data = pickle.dumps(data)
clientSocket.sendall(data)
print('Client sent model data to the server')

received_data = b''
while str(received_data)[-2] != '.':
	data = clientSocket.recv(8)
	received_data += data

received_data = pickle.loads(received_data)
print("Received status from server: {data}".format(data=received_data))

clientSocket.close()
print('Client socket closed')

