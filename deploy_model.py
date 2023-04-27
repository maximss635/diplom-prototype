import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from tensorflow import keras

from utils import read_config, setup_logger


class HttpHandler(BaseHTTPRequestHandler):
    def __init__(self, *args):
        BaseHTTPRequestHandler.__init__(self, *args)

    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_POST(self):
        logging.info("New POST request")

        length = int(self.headers.get('content-length'))
        body = self.rfile.read(length)
        body = body.decode()
        body = json.loads(body)

        prediction = self.server.predict(body["x"])

        if prediction:
            answer = {"is_attack": True}
        else:
            answer = {"is_attack": False}

        self._set_headers()
        self.wfile.write(bytes(json.dumps(answer), encoding="utf-8"))


class HTTPModelServer(HTTPServer):
    def __init__(self):
        HTTPServer.__init__(self, ("", 8000), HttpHandler)

        logging.info("Running server")

        self._config = read_config()

        self.load_model()

    def load_model(self):
        logging.info("Loading model")
        self.model = keras.models.load_model(
            self._config["protection_methods"]\
                ["distillation"]\
                ["student_model"]\
                ["dir"]
        )

    def predict(self, x):
        return self.model.predict([x])[0][0] > 0.5


def main():
    setup_logger()
    HTTPModelServer().serve_forever()


if __name__ == "__main__":
    main()
