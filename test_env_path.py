import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

login = int(os.getenv('MT5_LOGIN', 0))
password = os.getenv('MT5_PASSWORD', '')
server = os.getenv('MT5_SERVER', '')
path = os.getenv('MT5_TERMINAL_PATH', '')

print(f"Testing MT5 Account: {login}")
print(f"Server: {server}")
print(f"Path: {path}")

# Specifically use the path from .env
res = mt5.initialize(
    path=path,
    login=login,
    password=password,
    server=server
)

if res:
    print(f"SUCCESS: Connected to MT5 Account {mt5.account_info().login}")
    mt5.shutdown()
else:
    print(f"FAILED: {mt5.last_error()}")
