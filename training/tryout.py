from time import sleep

import training.udp

tryout_port = 6000


def tryout_client():
    print("---> tryout client")
    training.udp.send_and_wait("hallo", tryout_port)


def tryout_server():
    def handler1(msg: str) -> str:
        sleep(0.5)
        return f"response to <{msg}>"

    print("---> tryout server")
    training.udp.open_socket(tryout_port, 5, handler1)


def main():
    # tryout_all_sims()
    tryout_client()
    # tryout_date()
