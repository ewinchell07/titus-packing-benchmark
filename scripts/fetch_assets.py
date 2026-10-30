"""Fetch the exact HF asset revision bound to this code release and verify it."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked_target(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or not relative.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    target = root.joinpath(*relative.parts)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root):
        raise ValueError(f'Unsafe symlink destination: {name}')
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Unsafe destination: {name}')
    return target


def install_archive(archive_path, spec, root):
    root = Path(root).resolve()
    if digest(archive_path) != spec['sha256']:
        raise ValueError('Downloaded archive checksum differs from assets.lock.json')
    expected = spec['files_sha256']
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix='titus-assets-') as temp:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive member list differs from assets.lock.json')
        staging = Path(temp)
        targets = {}
        # Verify every member and destination before installing any file.
        for info in archive.infolist():
            name = info.filename
            target = checked_target(root, name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f'Unsafe non-file archive member: {name}')
            staged = staging.joinpath(*PurePosixPath(name).parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, staged.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            if digest(staged) != expected[name]:
                raise ValueError(f'Archive member checksum differs: {name}')
            if target.exists() and (not target.is_file() or digest(target) != expected[name]):
                raise ValueError(f'Refusing to overwrite modified asset: {name}')
            targets[name] = target
        installed = 0
        for name, target in targets.items():
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging.joinpath(*PurePosixPath(name).parts), target)
            installed += 1
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('scene', 'runtime', 'all'), default='all')
    parser.add_argument('--offline-dir', type=Path, help='Use already downloaded release ZIPs')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'assets.lock.json').read_text())
    if not re.fullmatch(r'[0-9a-f]{40}', lock['revision'] or ''):
        raise ValueError('assets.lock.json must pin an immutable HF commit')
    selected = list(lock['components']) if args.component == 'all' else [args.component]
    for name in selected:
        spec = lock['components'][name]
        if args.offline_dir:
            archive = args.offline_dir / spec['filename']
        else:
            from huggingface_hub import hf_hub_download
            archive = hf_hub_download(repo_id=lock['repo_id'], repo_type='dataset',
                                      revision=lock['revision'], filename=spec['filename'])
        installed = install_archive(archive, spec, ROOT)
        print(json.dumps({'component': name, 'verified_files': len(spec['files_sha256']),
                          'installed_files': installed, 'revision': lock['revision']}))


if __name__ == '__main__':
    main()
