"""Fetch the pinned public checkpoints and verify every SHA-256."""

import argparse, hashlib, json, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
args = p.parse_args()
lock = json.loads(Path(__file__).with_name('model-lock.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


for item in lock['files']:
    target = args.directory / item['path']
    target.parent.mkdir(exist_ok=True, parents=True)
    if target.exists():
        if digest(target) != item['sha256']:
            raise RuntimeError('Existing checkpoint has wrong hash: ' + str(target))
        continue
    temporary = target.with_suffix(target.suffix + '.partial')
    url = 'https://huggingface.co/' + lock['model'] + '/resolve/' + lock['revision'] + '/' + item['path']
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as out:
        while chunk := source.read(1024 * 1024):
            out.write(chunk)
    if digest(temporary) != item['sha256']:
        raise RuntimeError('Downloaded checkpoint has wrong hash: ' + str(target))
    temporary.replace(target)
    print(str(target), flush=True)
(args.directory / 'receipt.json').write_text(json.dumps(lock, indent=2) + '\n')
