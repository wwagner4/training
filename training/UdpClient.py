import socket

host = "127.0.0.1"


def send_and_wait(data: str, port: int) -> str:
    print(f"---> Sendig: {data}")
    bytes_to_send = str.encode(data)
    server_address_port = (host, port)
    buffer_size = 1024
    my_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    my_socket.settimeout(2)
    try:
        my_socket.sendto(bytes_to_send, server_address_port)
        msg_from_server = my_socket.recvfrom(buffer_size)
        result = msg_from_server[0].decode("ascii")
        print(f"<--- Response from server {result}")
        return result
    finally:
        my_socket.close()
        print("---- Socket closed")


def open_socket(port: int, handler) -> None:
    # TODO
    pass
