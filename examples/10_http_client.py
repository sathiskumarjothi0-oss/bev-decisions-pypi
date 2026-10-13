"""Call a self-hosted Bev server from any language; here with Python's standard library.

    pip install "bev-decisions[serve]" && bev serve --port 8000      # in another terminal
    python examples/10_http_client.py
"""

import json
import urllib.request

body = {"state": "My app crashes on start",
        "questions": {"team": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": "payments", "tech": "bugs and crashes"}}}}
req = urllib.request.Request("http://127.0.0.1:8000/v1/decide", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req)))
