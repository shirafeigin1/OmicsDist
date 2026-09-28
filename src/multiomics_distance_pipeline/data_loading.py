from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import normalize_sample_id


@dataclass
class MultiOmicsData:
    metadata: pd.DataFrame
    microbiome: pd.DataFrame
    metabolome: pd.DataFrame
    micro_features: list
    metab_features: list
    sample_ids: list
    micro_available: pd.Series
    metab_available: pd.Series

    @property
    def micro(self):
        return self.microbiome[self.micro_features].to_numpy(float)

    @property
    def metab(self):
        return self.metabolome[self.metab_features].to_numpy(float)


def load_tables(metadata_path: Path, microbiome_path: Path, metabolome_path: Path):
    tables = [
        normalize_sample_id(pd.read_csv(p))
        for p in [metadata_path, microbiome_path, metabolome_path]
    ]
    for table in tables:
        if table.SampleID.duplicated().any() or table.SampleID.isin(["nan", ""]).any():
            raise ValueError("Sample IDs must be unique and nonempty.")
    metadata, micro, metab = tables
    ids = metadata.SampleID.tolist()
    if set(micro.SampleID) - set(ids) or set(metab.SampleID) - set(ids):
        raise ValueError("Omics contain sample IDs absent from metadata.")
    micro = micro.set_index("SampleID").reindex(ids).reset_index()
    metab = metab.set_index("SampleID").reindex(ids).reset_index()
    mf = [c for c in micro if c != "SampleID"]
    tf = [c for c in metab if c != "SampleID"]
    for table, features in [(micro, mf), (metab, tf)]:
        x = table[features].to_numpy(float)
        if np.isinf(x).any():
            raise ValueError("Infinite input values.")
        if np.any(np.isnan(x).any(1) & ~np.isnan(x).all(1)):
            raise ValueError("Input contains a partially missing omics block.")
    ma = micro[mf].notna().all(axis=1)
    ta = metab[tf].notna().all(axis=1)
    if (~ma & ~ta).any():
        raise ValueError("Samples missing both omics are unsupported.")
    return MultiOmicsData(metadata, micro, metab, mf, tf, ids, ma, ta)


def data_structure_summary(data):
    return pd.DataFrame(
        {
            "item": [
                "samples",
                "microbiome_features",
                "metabolome_features",
                "complete_pairs",
                "microbiome_missing",
                "metabolome_missing",
            ],
            "value": [
                len(data.sample_ids),
                len(data.micro_features),
                len(data.metab_features),
                int((data.micro_available & data.metab_available).sum()),
                int((~data.micro_available).sum()),
                int((~data.metab_available).sum()),
            ],
        }
    )
