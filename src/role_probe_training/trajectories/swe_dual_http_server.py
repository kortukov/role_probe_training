"""Local diagnostic httpbin endpoint accepting plain HTTP and TLS on one port."""
import socket
import ssl
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server
from httpbin import app

class DualServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        try:
            if connection.recv(1, socket.MSG_PEEK) == b"\x16":
                connection = context.wrap_socket(connection, server_side=True)
            return connection, address
        except BaseException:
            connection.close()
            raise

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('/opt/swe-http-fixture/local.crt', '/opt/swe-http-fixture/local.key')
make_server('127.0.0.1', 8765, app, server_class=DualServer,
            handler_class=WSGIRequestHandler).serve_forever()
