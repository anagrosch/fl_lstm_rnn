from socket import *
import sys
import pickle

def sendData(socket, msg):
	msg = pickle.dumps(msg)
	socket.sendall(msg)
	print('Client sent message to the server')

	received_data = b''
	while str(received_data)[-2] != '.':
		data = clientSocket.recv(8)
		received_data += data

	return pickle.loads(received_data)


serverIP = "192.168.1.60"
serverPort = 10800

filename = sys.argv[1]
with open(filename, 'r') as f:
	data = f.readlines()

clientSocket = socket(AF_INET, SOCK_STREAM) #IPv4, TCP
clientSocket.connect((serverIP, serverPort))
print('Connected to server')

# Send filename to server
received_data = sendData(clientSocket, filename)
print("Received status from server: {data}".format(data=received_data))

# Send file data to server
received_data = sendData(clientSocket, data)
print("Received status from server: {data}".format(data=received_data))

clientSocket.close()
print('Client socket closed')

