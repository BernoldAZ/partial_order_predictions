"""Split BPI Challenge 2018.xes into one event log per document type.

The raw log has `application` as the case id, with every event carrying
`docid` / `doctype` attributes identifying the underlying document it
belongs to. Per the official data page
(https://ais.win.tue.nl/bpi/2018/challenge.html), the challenge organizers
also publish separate "document logs": one file per document type, with the
case id shifted from `application` to `docid`. This script reproduces that
split from the single raw file.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from pm4py.objects.conversion.log import converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def load_log(path):
    """Load an XES file into a flat pm4py DataFrame."""
    log = xes_importer.apply(path)
    df = converter.apply(log, variant=converter.Variants.TO_DATA_FRAME)
    return log, df


def _export_doctype(doctype, sub_df, extensions, classifiers, attributes, out_dir):
    """Convert one doctype's DataFrame slice to XES and write it to disk.

    Runs in a worker process, so it only receives the (much smaller)
    per-doctype slice and the source log's lightweight metadata, not the
    full source log.
    """
    event_log = converter.apply(sub_df, variant=converter.Variants.TO_EVENT_LOG)
    event_log.extensions.update(extensions)
    event_log.classifiers.update(classifiers)
    event_log.attributes.update(attributes)

    out_path = os.path.join(out_dir, f"BPIC18_{doctype.replace(' ', '_')}.xes.gz")
    xes_exporter.apply(event_log, out_path, parameters={'compress': True})
    return doctype, sub_df['case:concept:name'].nunique(), len(sub_df), out_path


def split_by_doctype(df, source_log, out_dir):
    """Write one gzip-compressed XES file per `doctype` value in *df*.

    Case id in each output file is `docid`; the originating application id
    is kept as the `application` trace attribute. One file is exported per
    worker process, in parallel.
    """
    os.makedirs(out_dir, exist_ok=True)
    event_cols = [c for c in df.columns if not c.startswith('case:')]

    doctype_frames = {}
    for doctype in sorted(df['doctype'].dropna().unique()):
        sub = df.loc[df['doctype'] == doctype, event_cols + ['case:concept:name']].copy()
        sub = sub.rename(columns={'case:concept:name': 'case:application'})
        sub['case:concept:name'] = sub['docid']
        sub = sub.sort_values(['case:concept:name', 'time:timestamp'])
        doctype_frames[doctype] = sub

    max_workers = min(len(doctype_frames), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_export_doctype, doctype, sub, source_log.extensions,
                             source_log.classifiers, source_log.attributes, out_dir)
            for doctype, sub in doctype_frames.items()
        ]
        for future in as_completed(futures):
            doctype, n_docs, n_events, out_path = future.result()
            print(f"{doctype}: {n_docs} documents, {n_events} events -> {out_path}")


def verify_split(out_dir, source_log_path):
    """Sanity-check the files written by split_by_doctype(). Run by hand."""
    _, source_df = load_log(source_log_path)
    expected_events = source_df['doctype'].notna().sum()

    total_events = 0
    all_ok = True
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith('.xes.gz'):
            continue
        expected_doctype = fname[len('BPIC18_'):-len('.xes.gz')].replace('_', ' ')
        log = xes_importer.apply(os.path.join(out_dir, fname))

        file_ok = True
        for trace in log:
            case_id = trace.attributes.get('concept:name')
            for event in trace:
                total_events += 1
                if event.get('doctype') != expected_doctype:
                    file_ok = False
                    print(f"  FAIL [{fname}] event doctype "
                          f"'{event.get('doctype')}' != '{expected_doctype}'")
                if event.get('docid') != case_id:
                    file_ok = False
                    print(f"  FAIL [{fname}] event docid '{event.get('docid')}' "
                          f"!= trace case id '{case_id}'")

        print(f"{fname}: {'OK' if file_ok else 'FAIL'}")
        all_ok = all_ok and file_ok

    print(f"Total events in output files: {total_events}")
    print(f"Events with a doctype in source log: {expected_events}")
    counts_ok = total_events == expected_events
    print(f"Event count match: {'OK' if counts_ok else 'FAIL'}")

    print(f"Overall: {'PASS' if (all_ok and counts_ok) else 'FAIL'}")


if __name__ == "__main__":
    log_path = os.path.join(os.path.dirname(__file__), "BPI Challenge 2018.xes")
    out_dir = os.path.join(os.path.dirname(__file__), "BPI Challenge 2018_document_logs")

    source_log, df = load_log(log_path)
    print(df['doctype'].value_counts())
    split_by_doctype(df, source_log, out_dir)
