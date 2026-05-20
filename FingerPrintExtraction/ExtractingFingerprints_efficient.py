"""
ExtractingFingerprints_efficient.py — memory-efficient FCFP6 add-on.

Standalone helper (does NOT touch ExtractingFingerprints.py). Streams the input
parquet in batches, generates an FCFP6 fingerprint per row from the SMILES
column, and writes the result to the output parquet with the new FCFP6 column
attached. Never loads the whole input file into memory at once, so it works on
460k+-row files where the regular script OOMs.

Output schema = input schema + 'FCFP6' (comma-separated bit string, same format
the rest of the pipeline expects). The existing fingerprint columns in the
input file are copied through verbatim (batch by batch) — they're never all
held in memory simultaneously.

Run:
  python -m FingerPrintExtraction.ExtractingFingerprints_efficient
"""
import pyarrow as pa
import pyarrow.parquet as pq

from .fingerprints import HitGenFCFP6


def _smiles_to_fp_string(smi, fp):
    """Generate one FCFP6 fingerprint as a comma-separated bit string.
    Matches the format the rest of the pipeline already uses for stored
    fingerprints; on failure returns 'nan,nan,...'."""
    try:
        arr = fp.generate_fps(smis=[smi]).flatten()
        return ','.join(map(str, arr))
    except Exception:
        return ','.join(['nan'] * fp._dimension)


def add_fcfp6_streaming(input_file, output_file,
                        smiles_column='SMILES',
                        new_column='FCFP6',
                        batch_size=10000):
    """Stream input → output, attaching the new FCFP6 column per batch.

    Peak memory ≈ one batch's worth of rows (and the FCFP6 strings for that
    batch), regardless of total file size.
    """
    fp = HitGenFCFP6()
    pf = pq.ParquetFile(input_file)
    total_rows = pf.metadata.num_rows

    print(f"[ExtractingFingerprints_efficient] Input  : {input_file}")
    print(f"[ExtractingFingerprints_efficient] Output : {output_file}")
    print(f"[ExtractingFingerprints_efficient] Adding : '{new_column}' from '{smiles_column}'")
    print(f"[ExtractingFingerprints_efficient] Batch  : {batch_size} rows")
    print(f"[ExtractingFingerprints_efficient] Total  : {total_rows} rows in {pf.num_row_groups} row groups")

    if smiles_column not in pf.schema_arrow.names:
        raise KeyError(f"Column '{smiles_column}' not found in {input_file}.")
    if new_column in pf.schema_arrow.names:
        raise ValueError(
            f"Column '{new_column}' already exists in {input_file}; refusing to overwrite. "
            "Pick a different new_column name or drop it from the source first."
        )

    writer = None
    processed = 0
    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
            # Each batch is a pyarrow.RecordBatch with the full input schema.
            smiles_list = batch.column(smiles_column).to_pylist()
            fp_strings = [_smiles_to_fp_string(smi, fp) for smi in smiles_list]
            new_col_array = pa.array(fp_strings, type=pa.string())
            new_batch = batch.append_column(new_column, new_col_array)

            if writer is None:
                writer = pq.ParquetWriter(output_file, new_batch.schema)
            writer.write_table(pa.Table.from_batches([new_batch]))

            processed += batch.num_rows
            print(f"[ExtractingFingerprints_efficient] Batch {batch_idx + 1}: "
                  f"wrote {batch.num_rows} rows; cumulative {processed}/{total_rows}")
    finally:
        if writer is not None:
            writer.close()

    print(f"[ExtractingFingerprints_efficient] Done — wrote {processed} rows to {output_file}")


def main():
    input_file = r"Data.parquet" # Parquet file
    output_file = r"Data_FPs.parquet"
    add_fcfp6_streaming(
        input_file=input_file,
        output_file=output_file,
        smiles_column='SMILES',
        new_column='FCFP6',
        batch_size=10000,
    )


if __name__ == "__main__":
    main()
