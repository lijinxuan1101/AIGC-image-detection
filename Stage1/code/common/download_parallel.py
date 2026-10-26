#!/usr/bin/env python3
"""多线程分块下载（HTTP Range），支持断点续传与 MD5 校验。

Zenodo 对单连接限速，多连接速度可叠加。每个分块完成后写入 `<out>.done` 记录，
中断后重跑同一命令即可续传。

用法：
  python download_parallel.py URL OUT --size BYTES --md5 HEX [--threads 32] [--chunk-mb 64]
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_chunk(url, out, start, end, retries=20):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"期望 206，得到 {r.status}")
                pos = start
                with open(out, "r+b") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.seek(pos)
                        f.write(buf)
                        pos += len(buf)
                if pos != end + 1:
                    raise RuntimeError(f"分块不完整 {pos}!={end + 1}")
            return
        except Exception as e:  # noqa: BLE001
            time.sleep(min(2 + attempt * 2, 30))
            last = e
    raise RuntimeError(f"分块 {start}-{end} 失败: {last}")


def md5_file(path, bufsize=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--md5", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--chunk-mb", type=int, default=64)
    a = ap.parse_args()

    chunk = a.chunk_mb << 20
    chunks = [(s, min(s + chunk, a.size) - 1) for s in range(0, a.size, chunk)]
    state_path = a.out + ".done"
    done = set()
    if os.path.exists(state_path):
        done = set(json.load(open(state_path)))
    if not os.path.exists(a.out) or os.path.getsize(a.out) != a.size:
        with open(a.out, "wb") as f:
            f.truncate(a.size)
        done = set()
    todo = [c for c in chunks if c[0] not in done]
    print(f"总分块 {len(chunks)}，已完成 {len(done)}，待下载 {len(todo)}", flush=True)

    lock = threading.Lock()
    t0 = time.time()
    got = [sum(min(s + chunk, a.size) - s for s in done)]

    def work(c):
        fetch_chunk(a.url, a.out, c[0], c[1])
        with lock:
            done.add(c[0])
            got[0] += c[1] - c[0] + 1
            json.dump(sorted(done), open(state_path, "w"))
        return c

    base = got[0]
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(work, c) for c in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            el = time.time() - t0
            rate = (got[0] - base) / el / 1e6
            print(f"[{i}/{len(todo)}] {got[0] / a.size * 100:.1f}%  {rate:.2f} MB/s", flush=True)

    print("下载完成，校验 MD5 ...", flush=True)
    actual = md5_file(a.out)
    ok = actual == a.md5
    print(f"md5 actual={actual} expected={a.md5} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    with open(a.out + ".md5check", "w") as f:
        f.write(f"{actual}  {os.path.basename(a.out)}\nexpected={a.md5}\nmatch={ok}\n")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
