import os
import json
import socket
import subprocess

DEFAULT_SOCKET_PATH = "/var/run/suricata/suricata-command.socket"

class SuricataSocketConnector:
    """
    Optimized connector for interacting directly with the Suricata Unix Socket
    to trigger non-blocking ruleset reloads. This reduces latency by eliminating 
    the subprocess overhead of launching 'suricatasc' on every domain detection.
    """
    def __init__(self, socket_path=DEFAULT_SOCKET_PATH, mock=False):
        self.socket_path = socket_path
        self.mock = mock

    def reload_ruleset(self) -> bool:
        """
        Triggers a non-blocking ruleset reload on Suricata.
        Tries direct Unix socket communication, falling back to subprocess if necessary.
        """
        if self.mock:
            print("[Mock] Suricata ruleset reload triggered successfully.")
            return True

        # Check if Unix sockets are supported (i.e. not Windows)
        if not hasattr(socket, "AF_UNIX"):
            print("[-] Direct Unix socket communication is not supported on this operating system.")
            return self._fallback_reload()

        # 1. Attempt direct connection to the Suricata Unix domain socket
        try:
            if not os.path.exists(self.socket_path):
                print(f"[-] Suricata socket not found at {self.socket_path}")
                return self._fallback_reload()

            print(f"[*] Connecting directly to Suricata socket: {self.socket_path}...")
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.socket_path)

            # JSON-RPC protocol message for Suricata command socket
            payload = {
                "version": "0.2.0",
                "command": "ruleset-reload-nonblocking"
            }
            
            # Send message with newline termination (Suricata socket protocol requirement)
            client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            
            # Receive response
            response_data = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in chunk:
                    break
            
            client.close()
            
            response = json.loads(response_data.decode("utf-8"))
            if response.get("return") == "OK":
                print("[+] Ruleset reload triggered via direct Unix socket successfully.")
                return True
            else:
                print(f"[-] Suricata socket returned error: {response}")
                return self._fallback_reload()
                
        except Exception as e:
            print(f"[-] Failed direct socket reload: {e}")
            return self._fallback_reload()

    def _fallback_reload(self) -> bool:
        """
        Fallback reload using the 'suricatasc' control utility in a subprocess.
        """
        print("[*] Falling back to suricatasc command line utility...")
        try:
            # -c runs the command, and we run 'ruleset-reload-nonblocking'
            result = subprocess.run(
                ["suricatasc", "-c", "ruleset-reload-nonblocking"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[+] Fallback reload succeeded: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] Fallback reload failed with exit code {e.returncode}: {e.stderr.strip()}")
            return False
        except FileNotFoundError:
            print("[-] 'suricatasc' executable not found. Rule reload skipped.")
            return False
        except Exception as e:
            print(f"[-] Fallback reload failed due to an unexpected error: {e}")
            return False

if __name__ == "__main__":
    # Test connector in mock mode
    connector = SuricataSocketConnector(mock=True)
    connector.reload_ruleset()
