from multiprocessing.connection import Listener

address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
listener = Listener(address, authkey=bytes("pw", encoding= 'utf-8'))
cache = {}
while True:
    print("Waiting for connection...")
    conn = listener.accept()
    print(f"Connection accepted from {listener.last_accepted}")
    while True:
        msg = conn.recv()
        # do something with msg
        if msg == 'close':
            print(f"Closing connection and waiting for new one.")
            conn.close()
            break
        # print(str(msg))
        if "data" in msg.keys():
            cache[msg["key"]] = msg["data"]
            print(f"Added data for key {msg['key']} to cache")
            conn.send(f"Added data for key {msg['key']} to cache")
        else:
            if msg["key"] in cache.keys():
                print(f"Found data for key {msg['key']} in cache")
                conn.send(cache[msg["key"]])
            else:
                print(f"Did not find data for key {msg['key']} in cache")
                conn.send(None)
listener.close()

