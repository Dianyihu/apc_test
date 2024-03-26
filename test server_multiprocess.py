# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/3/23 
@time: 18:39
'''


import asyncio
import socket

loop = asyncio.get_event_loop()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
server_socket.setblocking(False)
ip_port = ('0.0.0.0', 8000)
server_socket.bind(ip_port)
server_socket.listen(3)


def accept_conn(ser_sock):
    sock, addr = ser_sock.accept()
    print("receive connection from %s" % (addr,))
    loop.add_reader(sock, receive_data, sock)


def receive_data(sock):
    data = sock.recv(1024)
    print("receive:", data.decode("utf-8"))
    msg = b"echo:" + data
    loop.add_writer(sock, send_data, sock, msg)


def send_data(sock, msg):
    sock.send(msg)
    loop.remove_writer(sock)
    loop.remove_reader(sock)


loop.add_reader(server_socket, accept_conn, server_socket)

loop.run_forever()
