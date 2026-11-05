#!/bin/bash
# pull raw results from the Colab VMs into results/raw
set -e
mkdir -p /tmp/sync results/raw
for v in "$@"; do
  timeout 120 colab exec -s $v -f /tmp/tarres.py > /dev/null 2>&1 || true
  rm -f /tmp/sync/$v.tgz
  timeout 200 colab download -s $v /content/res.tgz /tmp/sync/$v.tgz > /dev/null 2>&1 && tar xzf /tmp/sync/$v.tgz -C results/ 2>/dev/null && echo "synced $v"
done
ls results/raw/*.json | wc -l
