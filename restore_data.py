#!/usr/bin/env python3
"""Restore losslessly compressed release data using only the Python standard library."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
BLOCK = 4 * 1024 * 1024

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()

def local_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside release: {name}')
    return path

def restore(record, verify_only=False, force=False):
    target = local_path(record['path'])
    if not verify_only and target.exists():
        if target.stat().st_size == record['bytes'] and digest(target) == record['sha256']:
            print(f"Already verified: {record['path']}", flush=True)
            return
        if not force:
            raise ValueError(f'Existing file differs: {target}; use --force to replace')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = None
    try:
        with tempfile.TemporaryFile() as compressed:
            for part in record['parts']:
                source = local_path(part['path'])
                h = hashlib.sha256()
                n = 0
                with source.open('rb') as stream:
                    for block in iter(lambda: stream.read(BLOCK), b''):
                        h.update(block)
                        n += len(block)
                        compressed.write(block)
                if n != part['bytes'] or h.hexdigest() != part['sha256']:
                    raise ValueError(f'Archive checksum mismatch: {source}')
            compressed.seek(0)
            h = hashlib.sha256()
            n = 0
            output = None
            if not verify_only:
                output = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
                temporary_output = Path(output.name)
            try:
                with gzip.GzipFile(fileobj=compressed, mode='rb') as stream:
                    for block in iter(lambda: stream.read(BLOCK), b''):
                        h.update(block)
                        n += len(block)
                        if output is not None:
                            output.write(block)
            finally:
                if output is not None:
                    output.close()
            if n != record['bytes'] or h.hexdigest() != record['sha256']:
                raise ValueError(f'Restored checksum mismatch: {target}')
            if temporary_output is not None:
                os.replace(temporary_output, target)
                temporary_output = None
        print(f"{'Verified archive' if verify_only else 'Restored'}: {record['path']}", flush=True)
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify decompressed data without writing it')
    parser.add_argument('--force', action='store_true', help='Replace existing files with differing checksums')
    args = parser.parse_args()
    records = json.loads((ROOT / 'data_archives/manifest.json').read_text())
    for record in records:
        restore(record, args.verify_only, args.force)
    print(f'All {len(records)} data archives verified.', flush=True)

if __name__ == '__main__':
    main()
