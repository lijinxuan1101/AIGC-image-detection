#!/usr/bin/env python3
"""多连接并行 Range 下载，按顺序把字节写到 stdout（用于 `| tar -xz` 边下边解压，不落盘整包）。

用法：python stream_http.py URL SIZE [--threads 16] [--chunk-mb 32] | tar -xz ...
"""
import argparse
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def fetch(url, s, e, retries=30):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != e - s + 1:
                raise RuntimeError("short read")
            return data
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(min(2 + 2 * k, 20))
    raise RuntimeError(f"chunk {s}-{e} failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("size", type=int)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=32)
    a = ap.parse_args()
    ch = a.chunk_mb << 20
    ranges = [(s, min(s + ch, a.size) - 1) for s in range(0, a.size, ch)]
    out = sys.stdout.buffer
    with ThreadPoolExecutor(a.threads) as ex:
        q = deque()
        it = iter(ranges)
        for _ in range(a.threads * 2):
            r = next(it, None)
            if r:
                q.append(ex.submit(fetch, a.url, *r))
        done = 0
        while q:
            data = q.popleft().result()
            out.write(data)
            done += len(data)
            r = next(it, None)
            if r:
                q.append(ex.submit(fetch, a.url, *r))
            print(f"{done / 1e9:.2f} GB", file=sys.stderr, flush=True)
    out.flush()


if __name__ == "__main__":
    main()
