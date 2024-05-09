import select
import socket
from typing import Callable

host = "0.0.0.0"


def send_and_wait(data: str, port: int, timeout_sec: int) -> str:
    server_address_port = (host, port)
    buffer_size = 1024
    print(f"---> {port} Sendig: {data}")
    my_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    my_socket.settimeout(timeout_sec)
    bytes_to_send = str.encode(data)
    my_socket.sendto(bytes_to_send, server_address_port)
    msg_from_server = my_socket.recvfrom(buffer_size)
    return msg_from_server[0].decode("ascii")


def open_socket(port: int, timeout_sec: int, handler: Callable[[str], str]) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, port))
        print(f"---- Server {port} listening")
        active = True
        while active:
            print(f"---- Server {port} waitinng for request")
            readable, writable, exceptional = select.select([sock], [], [], timeout_sec)
            if readable:
                data, address = sock.recvfrom(1024)
                msg = data.decode("utf-8")
                print(f"---- Server {port} Received data: {msg} from {address}")
                resp = handler(msg)
                print(f"---- Server {port} Sending data: {resp} to {address}")
                bytes_to_send = str.encode(resp)
                sock.sendto(bytes_to_send, address)
            else:
                print(f"---- Server {port} No data received for {timeout_sec} s")
                active = False
        return f"--- Server {port} closed"
