import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/api/v1/predict/demo_classifier"


def one(_):
    inputs = [random.random() for _ in range(8)]
    payload = {"inputs": inputs}
    r = requests.post(URL, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    n = 200
    workers = 20
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, range(n)))
    elapsed = time.time() - start
    print(f"{n} requests in {elapsed:.2f}s -> {n/elapsed:.2f} req/s")
    print("Sample:", res[0])


if __name__ == "__main__":
    main()
