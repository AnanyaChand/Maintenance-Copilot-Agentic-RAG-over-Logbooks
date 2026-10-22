"""Private HTTP worker. stdout is a bounded transport envelope, never a UI log."""
import json
import socket
import sys
from urllib import request, error


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    try:
        job = json.load(sys.stdin)
        body = json.dumps(job["payload"]).encode("utf-8")
        req = request.Request(job["url"], data=body, method="POST", headers={
            "Authorization": "Bearer " + job["key"], "Content-Type": "application/json"})
        opener = request.build_opener(NoRedirect())
        with opener.open(req, timeout=job["timeout"]) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError
        data = json.loads(raw)
        choice = data["choices"][0]
        message = choice["message"]
        if (choice.get("finish_reason") not in {"stop", "tool_calls"}
                or message.get("refusal") or message.get("role") != "assistant"):
            raise ValueError
        print(json.dumps({"message": message}))
    except (TimeoutError, socket.timeout):
        print(json.dumps({"error": "timeout"}))
    except error.URLError as exc:
        print(json.dumps({"error": "timeout" if isinstance(exc.reason, TimeoutError) else "provider_error"}))
    except (ValueError, KeyError, IndexError, TypeError):
        print(json.dumps({"error": "malformed_output"}))
    except Exception:
        print(json.dumps({"error": "provider_error"}))


if __name__ == "__main__":
    main()
