"""Inspect a bounded set of actual HDF5 files without assuming a RoboTwin version."""
import argparse
import json
from pathlib import Path

from common import save


def inspect(path):
    import h5py
    rows = []
    with h5py.File(path, 'r') as f:
        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            low = name.lower()
            rows.append(dict(key=name, shape=list(obj.shape), dtype=str(obj.dtype),
                image_candidate=any(k in low for k in ['rgb', 'colors']),
                segmentation_candidate=any(k in low for k in ['segmentation', 'segment_id', 'actor_id', 'mesh_id']),
                timestamp_candidate='timestamp' in low))
        f.visititems(visitor)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, datasets=rows,
        note='Names/shapes only; verify semantic identity, channel order, version and raw ID encoding separately')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--max-files', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert 1 <= a.max_files <= 20
    if a.output.exists():
        raise FileExistsError(a.output)
    source = a.input.resolve()
    if source.is_dir() and a.output.resolve().is_relative_to(source):
        raise ValueError('Write inventory outside source dataset')
    if a.output.resolve() == source:
        raise ValueError('Output equals input')
    if source.is_file():
        paths = [source]
    else:
        paths = []
        for path in source.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.h5', '.hdf5'):
                paths.append(path)
                if len(paths) == a.max_files:
                    break
        paths.sort()
    if not paths:
        raise FileNotFoundError('No HDF5 found; inspect the actual alternative format instead of guessing')
    result = dict(status='inventory_only', selection='first filesystem-discovered bounded files, sorted; not a statistical sample',
        files=[inspect(path) for path in paths[:a.max_files]], raw_data_modified=False)
    save(a.output, result)
    print(json.dumps(dict(files=len(result['files']), output=str(a.output)), indent=2))


if __name__ == '__main__':
    main()
