"""
ExtractingFingerprints_efficient.py — memory-efficient streaming fingerprint extractor.

Standalone helper (does NOT touch ExtractingFingerprints.py). Streams the input
in batches, generates the requested fingerprints per row from the SMILES column,
and writes the result with the new fingerprint columns attached. Never loads
the whole input file into memory at once, so it works on 460k+-row files where
the regular script OOMs.

Output schema = input schema + each requested fingerprint column (comma-separated
bit string, same format the rest of the pipeline expects). Existing columns in
the input file are copied through verbatim (batch by batch).

Supports both .parquet and .csv input/output (auto-detected from the extension).
Output extension dictates the writer; you can mix formats (e.g. .csv → .parquet).

Run:
  python -m FingerPrintExtraction.ExtractingFingerprints_efficient
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .fingerprints import (
    HitGenMACCS, HitGenECFP4, HitGenECFP6, HitGenFCFP4, HitGenFCFP6,
    HitGenRDK, HitGenAvalon, HitGenTopTor, HitGenAtomPair,
)


# Catalogue of every supported fingerprint name → generator class. Pass any
# subset as the `fingerprints` dict in add_fingerprints_streaming().
FP_CATALOGUE = {
    'ECFP4':    HitGenECFP4,
    'ECFP6':    HitGenECFP6,
    'FCFP4':    HitGenFCFP4,
    'FCFP6':    HitGenFCFP6,
    'MACCS':    HitGenMACCS,
    'RDK':      HitGenRDK,
    'AVALON':   HitGenAvalon,
    'TOPTOR':   HitGenTopTor,
    'ATOMPAIR': HitGenAtomPair,
}


def _smiles_to_fp_string(smi, fp):
    """Generate one fingerprint as a comma-separated bit string.
    Matches the format the rest of the pipeline already uses for stored
    fingerprints; on failure returns 'nan,nan,...'."""
    try:
        arr = fp.generate_fps(smis=[smi]).flatten()
        return ','.join(map(str, arr))
    except Exception:
        return ','.join(['nan'] * fp._dimension)


def _make_fp_dict(names):
    """Convert a list/tuple of fingerprint names into a dict of {name: instance}."""
    instances = {}
    for n in names:
        if n not in FP_CATALOGUE:
            raise ValueError(
                f"Unknown fingerprint '{n}'. Valid names: {sorted(FP_CATALOGUE.keys())}"
            )
        instances[n] = FP_CATALOGUE[n]()
    return instances


def _input_is_parquet(path):
    return path.lower().endswith('.parquet')


def _output_is_parquet(path):
    return path.lower().endswith('.parquet')


def add_fingerprints_streaming(input_file, output_file,
                               fingerprints,
                               smiles_column='SMILES',
                               batch_size=10000):
    """Stream input → output, attaching the requested fingerprint columns per batch.

    Parameters
    ----------
    fingerprints : dict | list | tuple
        Either a list of fingerprint NAMES (e.g. ['ECFP4', 'FCFP4', 'ATOMPAIR']) —
        the function instantiates each from FP_CATALOGUE — OR a dict of
        {output_column_name: fingerprint_instance} if you need custom column
        names or already-instantiated generators.
    smiles_column : str
        Column in the input file holding SMILES strings.
    batch_size : int
        Number of rows processed per chunk. Lower → less RAM, more I/O overhead.
    """
    if isinstance(fingerprints, (list, tuple)):
        fingerprints = _make_fp_dict(fingerprints)
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("`fingerprints` must be a non-empty list/tuple of names or dict of {name: instance}.")

    new_columns = list(fingerprints.keys())

    print(f"[ExtractingFingerprints_efficient] Input  : {input_file}")
    print(f"[ExtractingFingerprints_efficient] Output : {output_file}")
    print(f"[ExtractingFingerprints_efficient] Adding : {new_columns}  from '{smiles_column}'")
    print(f"[ExtractingFingerprints_efficient] Batch  : {batch_size} rows")

    if _input_is_parquet(input_file):
        _run_parquet_input(input_file, output_file, fingerprints, smiles_column, batch_size)
    elif input_file.lower().endswith('.csv'):
        _run_csv_input(input_file, output_file, fingerprints, smiles_column, batch_size)
    else:
        raise ValueError(f"Unsupported input format: {input_file} (use .parquet or .csv)")

    print(f"[ExtractingFingerprints_efficient] Done -> {output_file}")


def _run_parquet_input(input_file, output_file, fingerprints, smiles_column, batch_size):
    """Stream parquet input via pyarrow batches. Output may be .parquet or .csv."""
    pf = pq.ParquetFile(input_file)
    total_rows = pf.metadata.num_rows
    schema_names = pf.schema_arrow.names

    if smiles_column not in schema_names:
        raise KeyError(f"Column '{smiles_column}' not found in {input_file}.")
    clash = [c for c in fingerprints if c in schema_names]
    if clash:
        raise ValueError(
            f"Columns {clash} already exist in {input_file}; refusing to overwrite. "
            "Rename the requested output columns or drop them from the source first."
        )

    parquet_writer = None
    csv_header_written = False
    processed = 0
    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
            smiles_list = batch.column(smiles_column).to_pylist()
            new_batch = batch
            for col_name, fp in fingerprints.items():
                fp_strings = [_smiles_to_fp_string(smi, fp) for smi in smiles_list]
                new_batch = new_batch.append_column(col_name, pa.array(fp_strings, type=pa.string()))

            if _output_is_parquet(output_file):
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_file, new_batch.schema)
                parquet_writer.write_table(pa.Table.from_batches([new_batch]))
            else:
                df_batch = new_batch.to_pandas()
                df_batch.to_csv(output_file, mode='a' if csv_header_written else 'w',
                                index=False, header=not csv_header_written)
                csv_header_written = True

            processed += batch.num_rows
            print(f"[ExtractingFingerprints_efficient] Batch {batch_idx + 1}: "
                  f"wrote {batch.num_rows} rows; cumulative {processed}/{total_rows}")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()


def _run_csv_input(input_file, output_file, fingerprints, smiles_column, batch_size):
    """Stream CSV input via pandas chunksize. Output may be .parquet or .csv."""
    # Peek at the header so we can validate before iterating.
    header = pd.read_csv(input_file, nrows=0).columns.tolist()
    if smiles_column not in header:
        raise KeyError(f"Column '{smiles_column}' not found in {input_file}.")
    clash = [c for c in fingerprints if c in header]
    if clash:
        raise ValueError(
            f"Columns {clash} already exist in {input_file}; refusing to overwrite. "
            "Rename the requested output columns or drop them from the source first."
        )

    parquet_writer = None
    csv_header_written = False
    processed = 0
    for batch_idx, df_batch in enumerate(pd.read_csv(input_file, chunksize=batch_size)):
        smiles_list = df_batch[smiles_column].tolist()
        for col_name, fp in fingerprints.items():
            df_batch[col_name] = [_smiles_to_fp_string(smi, fp) for smi in smiles_list]

        if _output_is_parquet(output_file):
            table = pa.Table.from_pandas(df_batch, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_file, table.schema)
            parquet_writer.write_table(table)
        else:
            df_batch.to_csv(output_file, mode='a' if csv_header_written else 'w',
                            index=False, header=not csv_header_written)
            csv_header_written = True

        processed += len(df_batch)
        print(f"[ExtractingFingerprints_efficient] Batch {batch_idx + 1}: "
              f"wrote {len(df_batch)} rows; cumulative {processed}")

    if parquet_writer is not None:
        parquet_writer.close()


# ---------------------------------------------------------------------------
# Backwards-compatible alias (older calls that passed a single new_column).
# ---------------------------------------------------------------------------
def add_fcfp6_streaming(input_file, output_file,
                        smiles_column='SMILES',
                        new_column='FCFP6',
                        batch_size=10000):
    """Legacy single-fingerprint wrapper. The column name MUST match a real
    fingerprint in FP_CATALOGUE — passing new_column='ECFP4' will now correctly
    generate ECFP4 (not silently emit FCFP6 with an ECFP4 label, like before)."""
    add_fingerprints_streaming(
        input_file=input_file,
        output_file=output_file,
        fingerprints=[new_column],
        smiles_column=smiles_column,
        batch_size=batch_size,
    )


def main():
    input_file  = r"D:\0000-UHN\PGK2\PGK2-Challange-splits\bitbirch\fold0.csv"
    output_file = r"D:\0000-UHN\PGK2\PGK2-Challange-splits\bitbirch\fold0_withfps.csv"

    # Pick any subset of FP_CATALOGUE keys — order is preserved in the output schema.
    fingerprints_to_add = ['ECFP4', 'FCFP4', 'ATOMPAIR', 'TOPTOR']

    add_fingerprints_streaming(
        input_file=input_file,
        output_file=output_file,
        fingerprints=fingerprints_to_add,
        smiles_column='SMILES',
        batch_size=10000,
    )


if __name__ == "__main__":
    main()
