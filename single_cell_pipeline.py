import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvi  
import diffxpy.api as de
import gseapy as gp

# IO and Utility

def read_counts_csv(csv_path: str) -> sc.AnnData:
    """Read a counts CSV (genes x cells or cells x genes) and return AnnData (cells x genes)."""
    adata = sc.read_csv(csv_path).T
    adata.obs["Sample"] = infer_sample_name_from_path(csv_path)
    return adata


def infer_sample_name_from_path(path: str) -> str:
    """
    Infer sample name from filename used in the original notebook:
    Expecting 'raw_counts/GSM5226574_C51ctr_raw_counts.csv' -> 'C51ctr' (2nd underscore token).
    Falls back to stem if pattern doesn't match.
    """
    fname = os.path.basename(path)
    toks = fname.split("_")
    if len(toks) >= 2:
        return toks[1]  
    return os.path.splitext(fname)[0]


def get_ribosome_genes(ribo_source: str | None = None) -> pd.Series:
    """
    Retrieve ribosomal gene list.
    - If ribo_source is a path to a local TXT, read first column.
    - If ribo_source is a valid HTTP/HTTPS URL, attempt to download (requires internet).
    - If None or fails, return empty Series (ribosomal filter will be skipped).
    """
    if ribo_source is None:
        return pd.Series(dtype=object)

    try:
        if ribo_source.lower().startswith(("http://", "https://")):
            df = pd.read_table(ribo_source, skiprows=2, header=None)
            return df[0]
        else:
            df = pd.read_table(ribo_source, header=None)
            return df[0]
    except Exception as e:
        print(f"[WARN] Could not load ribosomal genes from {ribo_source}: {e}", file=sys.stderr)
        return pd.Series(dtype=object)


# Doublet Removal (SOLO/scvi)

def run_solo_doublet_detection(adata: sc.AnnData, n_top_genes: int = 2000) -> pd.DataFrame:
    """
    Train SCVI + SOLO on the given AnnData and return per-cell doublet predictions.
    Adds highly variable gene selection before model training.
    """
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True, flavor="seurat_v3")

    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train()

    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train()

    df = solo.predict()
    df["prediction"] = solo.predict(soft=False)  # 'doublet' / 'singlet'

    # removed fragile barcode trimming that could break index matching
    # df.index = df.index.map(lambda x: x[:-2])

    df["dif"] = df.doublet - df.singlet
    return df


def filter_doublets(adata: sc.AnnData, solo_df: pd.DataFrame, dif_threshold: float = 0.5) -> sc.AnnData:
    """Return AnnData filtered to exclude predicted doublets using a confidence threshold on (doublet - singlet)."""
    # Threshold must be <= 1.0; use a practical default (0.5) and '>='
    doublets = solo_df[(solo_df["prediction"] == "doublet") & (solo_df["dif"] >= dif_threshold)]
    adata.obs["doublet"] = adata.obs.index.isin(doublets.index)
    return adata[~adata.obs["doublet"]].copy()


# QC and Preprocessing

def add_qc_metrics(
    adata: sc.AnnData,
    ribo_genes: Optional[pd.Series] = None,
    mt_prefixes: List[str] = ("MT-", "mt-"),
) -> None:
    """Annotate mitochondrial and ribosomal genes; compute Scanpy QC metrics."""
    # mark mitochondrial genes
    is_mt = np.zeros(adata.var_names.size, dtype=bool)
    for p in mt_prefixes:
        is_mt |= adata.var_names.str.startswith(p)
    adata.var["mt"] = is_mt

    # ribosomal genes (optional)
    if ribo_genes is not None and len(ribo_genes) > 0:
        adata.var["ribo"] = adata.var_names.isin(set(ribo_genes.astype(str).values))
        qc_vars = ["mt", "ribo"]
    else:
        adata.var["ribo"] = False
        qc_vars = ["mt"]

    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, percent_top=None, log1p=False, inplace=True)


def qc_filter(
    adata: sc.AnnData,
    min_cells_per_gene: int = 3,
    max_pct_mt: float = 20.0,
    max_pct_ribo: Optional[float] = 2.0,
    n_genes_high_quantile: float = 0.98,
) -> sc.AnnData:
    """
    Apply QC filters similar to the notebook:
      - remove genes seen in < min_cells_per_gene cells
      - remove cells above 98th percentile of n_genes_by_counts
      - remove cells with high mitochondrial or ribosomal content
    """
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    upper_lim = np.quantile(adata.obs["n_genes_by_counts"].values, n_genes_high_quantile)
    adata = adata[adata.obs["n_genes_by_counts"] < upper_lim].copy()

    if max_pct_mt is not None:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()

    if (max_pct_ribo is not None) and ("pct_counts_ribo" in adata.obs):
        adata = adata[adata.obs["pct_counts_ribo"] < max_pct_ribo].copy()

    return adata


def normalize_and_log(adata: sc.AnnData, target_sum: float = 1e4) -> sc.AnnData:
    """Normalise counts to target_sum and log1p; store raw."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata
    return adata


def compute_hvg_scale_pca_neighbors_umap_leiden(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    regress_vars: Optional[List[str]] = ("total_counts", "pct_counts_mt", "pct_counts_ribo"),
    n_pcs: int = 30,
    leiden_resolution: float = 0.5,
) -> sc.AnnData:
    """Standard clustering workflow: HVG -> regress_out -> scale -> PCA -> neighbors -> UMAP -> Leiden."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()

    if regress_vars:
        present = [v for v in regress_vars if v in adata.obs.columns]
        if present:
            sc.pp.regress_out(adata, present)

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)
    return adata


# Per-sample preprocessing (used for integration)

def preprocess_one_sample(
    csv_path: str,
    ribo_genes: Optional[pd.Series] = None,
    solo_dif_threshold: float = 0.5,
) -> sc.AnnData:
    """
    Per-sample preprocessing with SOLO-based doublet removal and QC.
    Mirrors the notebook's 'pp' function but is more robust.
    """
    adata = read_counts_csv(csv_path)

    # Doublet removal
    solo_df = run_solo_doublet_detection(adata.copy())
    adata = filter_doublets(adata, solo_df, dif_threshold=solo_dif_threshold)

    # Cells & genes filters
    sc.pp.filter_cells(adata, min_genes=200)

    # QC annotations & metrics
    add_qc_metrics(adata, ribo_genes=ribo_genes)
    adata = qc_filter(adata)

    return adata


def integrate_samples_from_dir(raw_counts_dir: str, ribo_source: Optional[str] = None) -> sc.AnnData:
    """Load and preprocess all CSVs in a directory, then concatenate into one AnnData."""
    ribo_genes = get_ribosome_genes(ribo_source)
    outs = []
    for fname in sorted(os.listdir(raw_counts_dir)):
        low = fname.lower()
        # allow gzipped CSVs
        if not (low.endswith(".csv") or low.endswith(".csv.gz")):
            continue
        path = os.path.join(raw_counts_dir, fname)
        print(f"[INFO] Preprocessing sample: {path}")
        outs.append(preprocess_one_sample(path, ribo_genes=ribo_genes))

    if not outs:
        raise FileNotFoundError(f"No CSV/CSV.GZ files found in {raw_counts_dir}")

    adata = sc.concat(
        outs,
        join="outer",
        label="Sample",
        keys=[a.obs["Sample"].unique()[0] for a in outs]
    )
    sc.pp.filter_genes(adata, min_cells=10)

    # make counts matrix CSR as in notebook
    from scipy.sparse import csr_matrix
    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    return adata


# SCVI Integration

def scvi_integrate(
    adata: sc.AnnData,
    batch_key: str = "Sample",
    cont_covariates: Optional[List[str]] = ("pct_counts_mt", "total_counts", "pct_counts_ribo"),
    latent_key: str = "X_scVI",
    normalized_layer: str = "scvi_normalized",
    leiden_resolution: float = 0.5,
) -> tuple[sc.AnnData, "scvi.model.SCVI"]:
    """SCVI integration producing latent embedding and normalized expression layer."""
    scvi.settings.seed = 0
    np.random.seed(0)

    sc.pp.filter_genes(adata, min_cells=100)
    adata.layers["counts"] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        categorical_covariate_keys=[batch_key] if batch_key in adata.obs.columns else None,
        continuous_covariate_keys=[c for c in (cont_covariates or []) if c in adata.obs.columns] or None,
    )

    model = scvi.model.SCVI(adata)
    model.train()

    adata.obsm[latent_key] = model.get_latent_representation()
    # Store numpy array in layer
    adata.layers[normalized_layer] = model.get_normalized_expression(library_size=1e4, return_numpy=True)

    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)

    return adata, model


# Marker detection and labeling

def find_markers_scanpy(adata: sc.AnnData, group_key: str = "leiden", p_adj: float = 0.05, lfc: float = 0.5) -> pd.DataFrame:
    """Find cluster markers using Scanpy's rank_genes_groups and return filtered DataFrame."""
    sc.tl.rank_genes_groups(adata, groupby=group_key, method="wilcoxon")
    markers = sc.get.rank_genes_groups_df(adata, None)
    markers = markers[(markers["pvals_adj"] < p_adj) & (markers["logfoldchanges"] > lfc)]
    return markers


def find_markers_scvi(adata: sc.AnnData, model, group_key: str = "leiden", fdr: float = 0.05, lfc: float = 0.5) -> pd.DataFrame:
    """Find cluster markers using scvi-tools differential_expression per group vs rest."""
    df = model.differential_expression(groupby=group_key)
    # use the provided fdr threshold
    fdr_col = "is_de_fdr_0.05"
    if fdr != 0.05:
        # scvi returns bayes factors/fdr columns depending on version; fall back to p-value if needed
        # if custom FDR is required, user can filter on 'qval' when present
        if "qval" in df.columns:
            df = df[df["qval"] < fdr]
        else:
            df = df[df.get(fdr_col, True)]
    else:
        df = df[df.get(fdr_col, True)]
    df = df[df["lfc_mean"] > lfc]
    return df


def assign_cell_types(adata: sc.AnnData, mapping: Dict[str, str], cluster_key: str = "leiden", out_key: str = "cell_type") -> None:
    """Map cluster IDs to cell types in-place."""
    adata.obs[out_key] = adata.obs[cluster_key].map(mapping).astype("category")


# Analysis helpers (counts, DE, enrichment, signatures)

def add_condition_from_sample(adata: sc.AnnData, covid_token: str = "cov") -> None:
    """Add a 'condition' column based on Sample name containing a token (e.g., 'cov' -> COVID19 else control)."""
    def _map(x: str) -> str:
        return "COVID19" if (isinstance(x, str) and covid_token.lower() in x.lower()) else "control"
    adata.obs["condition"] = adata.obs["Sample"].map(_map)


def summarize_cell_type_frequencies(adata: sc.AnnData, cell_type_key: str = "cell_type") -> pd.DataFrame:
    """Compute per-sample cell type frequencies."""
    counts = adata.obs.groupby(["Sample", "condition", cell_type_key]).size().reset_index(name="count")

    total_per_sample = adata.obs.groupby("Sample").size().rename("total_cells")
    counts = counts.merge(total_per_sample, on="Sample").assign(frequency=lambda d: d["count"] / d["total_cells"])
    return counts


def de_wald_diffxpy(subset: sc.AnnData, formula: str = "~ 1 + cell_type", factor_to_test: str = "cell_type") -> pd.DataFrame:
    """Run diffxpy Wald test (convert sparse to dense as in notebook)."""
    if hasattr(subset.X, "toarray"):
        subset.X = subset.X.toarray()
    sc.pp.filter_genes(subset, min_cells=100)
    res = de.test.wald(data=subset, formula_loc=formula, factor_loc_totest=factor_to_test)
    return res.summary().sort_values("log2fc", ascending=False).reset_index(drop=True)


def de_scvi_pairwise(adata: sc.AnnData, model, idx1_mask, idx2_mask) -> pd.DataFrame:
    """Pairwise DE with scvi-tools given boolean masks."""
    # Pass masks directly, not wrapped in single-element lists
    df = model.differential_expression(idx1=idx1_mask, idx2=idx2_mask)
    df = df[(df.get("is_de_fdr_0.05", True)) & (np.abs(df["lfc_mean"]) > 0.5)]
    df = df[(df.get("raw_normalized_mean1", 0) > 0.5) | (df.get("raw_normalized_mean2", 0) > 0.5)]
    return df.sort_values("lfc_mean")


def enrich_gseapy(gene_list: List[str], background: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Run Enrichr via gseapy (requires internet). Returns results or None on failure."""
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2021"],
            organism="human",
            outdir=None,
            background=background,
        )
        return enr.results
    except Exception as e:
        print(f"[WARN] Enrichment analysis skipped due to error: {e}", file=sys.stderr)
        return None


def score_signature(adata: sc.AnnData, genes: List[str], score_name: str = "signature_score") -> None:
    """Score a user-provided gene signature and add to adata.obs[score_name]."""
    sc.tl.score_genes(adata, genes, score_name=score_name)


# Main pipeline

def main(args: argparse.Namespace) -> None:
    sc.settings.verbosity = 2
    sc.set_figure_params(figsize=(6, 5))

    # 1) Integrate samples (doublet removal, QC) and write combined
    print("[STEP 1] Loading & preprocessing all samples...")
    adata = integrate_samples_from_dir(args.raw_counts_dir, ribo_source=args.ribo_source)
    combined_path = os.path.join(args.outdir, "combined.h5ad")
    os.makedirs(args.outdir, exist_ok=True)
    adata.write_h5ad(combined_path)
    print(f"[OK] Wrote {combined_path}")

    # 2) SCVI integration
    print("[STEP 2] SCVI integration...")
    adata, model = scvi_integrate(
        adata,
        batch_key=args.batch_key,
        leiden_resolution=args.leiden_resolution,
    )
    integrated_path = os.path.join(args.outdir, "integrated.h5ad")
    adata.write_h5ad(integrated_path)

    # Save scvi model to a directory
    model_dir = os.path.join(args.outdir, "scvi_model")
    try:
        model.save(model_dir)
    except Exception as e:
        print(f"[WARN] Could not save SCVI model: {e}", file=sys.stderr)
    print(f"[OK] Wrote {integrated_path} and saved model to {model_dir}.")

    # 3) Markers & labeling
    print("[STEP 3] Marker detection and (optional) cell-type labeling...")
    markers_scanpy = find_markers_scanpy(adata, group_key=args.cluster_key)
    adata.uns["markers_scanpy"] = markers_scanpy

    markers_scvi = find_markers_scvi(adata, model, group_key=args.cluster_key)
    adata.uns["markers_scvi"] = markers_scvi

    if args.cell_type_map:
        try:
            mapping = json.loads(args.cell_type_map)
            if not isinstance(mapping, dict):
                raise ValueError("cell_type_map must be a JSON object")
            assign_cell_types(adata, mapping, cluster_key=args.cluster_key, out_key="cell_type")
        except Exception as e:
            print(f"[WARN] Could not parse/apply cell_type_map: {e}", file=sys.stderr)

    adata.write_h5ad(integrated_path)  

    # 4) Basic analysis: condition, counts/frequencies
    print("[STEP 4] Analysis: add condition and summarize frequencies...")
    add_condition_from_sample(adata, covid_token=args.covid_token)
    freq_df = summarize_cell_type_frequencies(adata, cell_type_key=args.cell_type_key)
    freq_csv = os.path.join(args.outdir, "cell_type_frequencies.csv")
    freq_df.to_csv(freq_csv, index=False)
    print(f"[OK] Wrote {freq_csv}")

    # 5) Differential expression examples (subset AT1 vs AT2 if present)
    print("[STEP 5] Differential expression examples...")
    if {"AT1", "AT2"}.issubset(set(adata.obs.get(args.cell_type_key, pd.Series(dtype=str)).astype(str))):
        subset = adata[adata.obs[args.cell_type_key].isin(["AT1", "AT2"])].copy()
        subset.obs = subset.obs.rename(columns={args.cell_type_key: "cell_type"})

        try:
            wald_df = de_wald_diffxpy(subset)
            wald_csv = os.path.join(args.outdir, "de_wald_AT1_vs_AT2.csv")
            wald_df.to_csv(wald_csv, index=False)
            print(f"[OK] Wrote {wald_csv}")
        except Exception as e:
            print(f"[WARN] diffxpy Wald DE skipped: {e}", file=sys.stderr)

        try:
            idx1 = adata.obs[args.cell_type_key].astype(str) == "AT1"
            idx2 = adata.obs[args.cell_type_key].astype(str) == "AT2"
            scvi_df = de_scvi_pairwise(adata, model, idx1_mask=idx1, idx2_mask=idx2)
            scvi_csv = os.path.join(args.outdir, "de_scvi_AT1_vs_AT2.csv")
            scvi_df.to_csv(scvi_csv)
            print(f"[OK] Wrote {scvi_csv}")
        except Exception as e:
            print(f"[WARN] scVI DE skipped: {e}", file=sys.stderr)
    else:
        print("[INFO] AT1/AT2 cell types not present; skipping DE examples.")

    # 6) Enrichment 
    print("[STEP 6] Enrichment analysis (optional)...")
    try:
        # Example: use upregulated genes from Wald if available
        if 'wald_df' in locals():
            up_genes = wald_df[wald_df["log2fc"] > 0]["gene"].dropna().astype(str).tolist()
            enr_df = enrich_gseapy(up_genes, background=adata.var_names.tolist())
            if enr_df is not None:
                enr_csv = os.path.join(args.outdir, "enrichr_results.csv")
                enr_df.to_csv(enr_csv, index=False)
                print(f"[OK] Wrote {enr_csv}")
    except Exception as e:
        print(f"[WARN] Enrichment skipped: {e}", file=sys.stderr)

    # 7) Signature scoring
    if args.signature_file and os.path.exists(args.signature_file):
        print("[STEP 7] Signature scoring...")
        with open(args.signature_file) as f:
            genes = [x.strip() for x in f if x.strip()]
        score_signature(adata, genes, score_name=args.signature_name)
        adata.write_h5ad(integrated_path)  # persist scored obs
        print(f"[OK] Signature '{args.signature_name}' added and saved into {integrated_path}")

    print("[DONE] Pipeline complete.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-cell RNA-seq pipeline (Scanpy + scvi-tools).")
    p.add_argument("--raw_counts_dir", type=str, default="raw_counts", help="Directory with input .csv or .csv.gz raw counts.")
    p.add_argument("--outdir", type=str, default=".", help="Output directory for artifacts.")
    p.add_argument("--ribo_source", type=str, default=None,
                   help="Path or URL to ribosomal genes list (txt). If omitted, ribosomal filtering is limited.")
    p.add_argument("--batch_key", type=str, default="Sample", help="Batch key in .obs for scvi-tools.")
    p.add_argument("--cluster_key", type=str, default="leiden", help="Cluster key for marker detection and labeling.")
    p.add_argument("--cell_type_map", type=str, default=None,
                   help='JSON mapping from cluster IDs to cell types, e.g. \'{"0":"Macrophage",...}\'.')
    p.add_argument("--cell_type_key", type=str, default="cell_type", help="Column with assigned cell types (if mapped).")
    p.add_argument("--leiden_resolution", type=float, default=0.5, help="Leiden resolution for clustering.")
    p.add_argument("--covid_token", type=str, default="cov", help="Substring to define COVID19 samples in names.")
    p.add_argument("--signature_file", type=str, default=None, help="Optional gene signature file (one gene per line).")
    p.add_argument("--signature_name", type=str, default="signature_score", help="Name for signature score column.")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
