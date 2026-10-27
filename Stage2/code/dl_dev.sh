#!/bin/bash
cd "$(dirname "$0")/../data"
python3 ../code/stream_http.py https://zenodo.org/api/records/14963880/files/RRDataset_test.tar.gz/content 20117869400 --threads 24 2>../logs/dl_dev_progress.log | tar -xz -T dev_members.txt --wildcards
echo TARDONE
