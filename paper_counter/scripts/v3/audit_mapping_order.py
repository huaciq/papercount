"""Independently recompute ordinal filename ranks, without relabeling any GT."""
from pathlib import Path
from collections import defaultdict
import json
import re
from audit_p0 import OUT, write_csv, write_json


def capture_key(path):
    m = re.search(r'Pic_(\d{4})_(\d{2})_(\d{2})_(\d{6})_', path.name)
    if not m:
        raise ValueError('No capture timestamp: ' + path.name)
    duplicate = re.search(r'\((\d+)\)$', path.stem)
    return (*m.groups(), int(duplicate[1]) if duplicate else 0, path.name)


def main():
    truth = json.loads((OUT / 'p0/GT_VERSION.json').read_text(encoding='utf-8'))
    tables = defaultdict(list)
    for key, e in truth['images'].items():
        if e['mapping_method'] == 'timestamp_order_then_duplicate_index':
            tables[e['source']].append((key, e))
    rows = []
    for source, entries in sorted(tables.items()):
        folder = Path(source).parent
        actual_images = sorted(folder.glob('*.bmp'), key=capture_key)
        assert len(actual_images) == len(entries), (source, 'image cardinality')
        for key, e in entries:
            rank = next(i + 1 for i, p in enumerate(actual_images) if p.name == Path(key).name)
            rows.append(dict(image=key, table=source, snapshot_source_line=e['source_line'],
                             filename_timestamp_rank=rank, ordinal_match=rank == e['source_line'],
                             semantic_mapping_status='assumption_not_human_filename_verified'))
    assert all(r['ordinal_match'] for r in rows), 'GT ordinal mapping differs from filenames'
    write_csv(OUT / 'p0/filename_order_audit.csv', rows)
    write_json(OUT / 'p0/filename_order_summary.json', dict(tables=len(tables), ordinal_images=len(rows),
               mismatch=0, human_filename_mapping_verified=False))
    print(f'Filename ranks: {len(rows)} images in {len(tables)} tables, zero mismatches; semantic mapping remains assumed')


if __name__ == '__main__':
    main()
