# Single-Cell RNA-seq Pipeline: Scanpy + scvi-tools (with doublet removal, QC, integration, and DE)

A reproducible, end-to-end pipeline for single-cell RNA-sequencing (scRNA-seq) analysis. It harnesses **Scanpy** for preprocessing and quality control and integrates **scvi-tools** (SCVI + SOLO) for batch-aware embedding, clustering, and marker detection.

This README walks you through the pipeline’s functionalities, its components, dependencies, usage, and its conceptual foundation grounded in deep generative modeling of single-cell transcriptomics.

---

## Motivation & Conceptual Framework

This pipeline builds on the probabilistic foundation laid by **scVI**—a scalable, deep generative model for single-cell data analysis. It effectively reduces noise and batch effects while supporting clustering, visualization, and differential expression in a unified framework ([PubMed][1]).

By integrating SOLO for doublet detection within the scVI ecosystem, this workflow enhances data purity before integration, ensuring more reliable downstream analyses.

---

## Key Features

1. **Flexible Input Support**

   * Reads `*.csv` and compressed `*.csv.gz` count matrices.
   * Auto-infers sample names from file paths (expects 2nd underscore token as sample identifier).

2. **Doublet Detection**

   * Uses **SOLO** built on top of a trained **SCVI** model.
   * Robust default threshold (`dif ≥ 0.5`) for classifying doublets.

3. **Quality Control & Filtering**

   * Filters genes and cells based on expression, mitochondrial (`mt`), and optional ribosomal (`ribo`) content.
   * Supports custom ribosomal gene lists via local TXT or URL (downloads if needed).

4. **Normalisation and Dimensionality Reduction**

   * Standard normalisation, log transformation, HVG selection, regression, scaling, PCA, neighbor graph, UMAP embedding, and Leiden clustering.

5. **Data Integration**

   * Leverages SCVI to learn a shared latent representation across samples.
   * Produces normalized expression layers compatible with downstream analysis.

6. **Marker and Differential Expression Analysis**

   * Identifies cluster markers using both Scanpy’s Wilcoxon test and scVI’s probabilistic DE.
   * Allows FDR filtering adjustments (beyond default 0.05).

7. **Optional Analyses**

   * **Cell-type assignment** via user-supplied JSON map.
   * **Per-sample cell type frequency summaries**.
   * **Differential expression comparisons** (e.g., AT1 vs AT2).
   * **Enrichment analyses** using Enrichr via `gseapy`.
   * **Gene signature scoring** using `tl.score_genes`.

8. **Reproducibility Enhancements**

   * Fixed random seed for SCVI and NumPy.
   * Robust handling of model saving (directory-based).

---

## Installation and Dependencies

This pipeline requires a Python environment with:

* `scanpy`
* `scvi-tools` (for SCVI and SOLO)
* `pandas`, `numpy`, `scipy`
* Optional dependencies, imported internally:

  * `diffxpy`
  * `gseapy`

Use `requirements.txt` or `environment.yml` to pin versions as needed.

---

## Usage Instructions

```bash
python sc_rna_pipeline.py \
  --raw_counts_dir path/to/raw_counts \
  --outdir path/to/output \
  --ribo_source path/to/ribo_genes.txt \
  --batch_key Sample \
  --cluster_key leiden \
  --cell_type_map '{"0":"Macrophage","1":"AT1",...}' \
  --covid_token cov \
  --signature_file path/to/signature.txt \
  --signature_name my_signature
```

### Pipeline Steps

1. **Preprocessing**: reads raw counts, removes doublets (SOLO), applies QC.
2. **Integration**: trains SCVI model, outputs latent embedding and normalized data.
3. **Clustering & Marker Detection**: runs Leiden clustering, identifies markers (Scanpy + scVI DE).
4. **Annotations & Analysis**: maps clusters to cell types, calculates frequencies, runs DE (AT1 vs AT2), performs enrichment and signature scoring if requested.

---

## Output Artifacts

* `combined.h5ad`: raw processed dataset (QC applied)
* `integrated.h5ad`: SCVI-integrated dataset with embeddings, clusters, and metadata
* `scvi_model/`: saved SCVI model directory
* `cell_type_frequencies.csv`: per-sample cell type proportions
* `de_wald_AT1_vs_AT2.csv`, `de_scvi_AT1_vs_AT2.csv`: differential expression outputs
* `enrichr_results.csv`: enrichment analysis results
* (Optional) Updated `.h5ad` files with signature scores

---

## Conceptual Context & References

* **Lopez et al. (2018)** introduced scVI—a variational autoencoder model offering batch-corrected latent embeddings and probabilistic DE (Genetics, 2018) ([PubMed][1]).
* This pipeline follows a similar philosophy: integrating probabilistic modeling (SCVI) with a clean QC/preprocessing framework (including SOLO for doublet filtering) to enhance reproducibility and analytical rigor.

---

## License and Citation

* **Lopez et al. (2018)**: *Deep generative modeling for single-cell transcriptomics* ([PubMed][1]).
* **Melms et al. (2021)**: *A molecular single-cell lung atlas of lethal COVID-19* — as a case study and scientific inspiration for focusing QC and cell-type frequency analysis in disease contexts ([Nature][2]).

[1]: https://pubmed.ncbi.nlm.nih.gov/30504886/?utm_source=chatgpt.com "Deep generative modeling for single-cell transcriptomics"
[2]: https://www.nature.com/articles/s41586-021-03569-1 "A molecular single-cell lung atlas of lethal COVID-19 | Nature"