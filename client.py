import http.client
import json


def main():
    URL = "localhost:8000"
    x = [
        0.01729294,
        -0.61783926,
        0.56029819,
        -0.15544544,
        -0.09968001,
        -0.17064575,
        -0.1023306,
        -0.06552555,
        -0.09266922,
        -0.06538644,
        -0.02371417,
        -0.0835073,
    ]

    request_body = {"x": x}

    print("Send:\n{}".format(json.dumps(request_body, indent=4)))

    conn = http.client.HTTPConnection(URL)
    conn.request("POST", "/", body=json.dumps(request_body))

    answer = conn.getresponse()
    answer = json.loads(answer.read().decode("utf-8"))
    print("Recv:")
    print(json.dumps(answer, indent=4))


if __name__ == "__main__":
    main()
