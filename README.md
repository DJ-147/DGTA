## DGTA: Integration algorithm for single-cell RNA-seq and ATAC-seq based on Graph Transformer

This work proposes DGTA, based on Graph Transformer for the integration of scRNA-seq and scATAC-seq data. A robust hybrid graph construction method is included inside.

### Datasets (10.5281/zenodo.15657341)

Human myocardial infarction: [data](https://cellxgene.cziscience.com/collections/8191c283-0816-424b-9b61-c3e1d6258a77), peak-by-cell matrix([part1](https://zenodo.org/records/6578553), [part2](https://zenodo.org/record/6578617))

Mouse colon: [GSE207308](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE207308)

Mouse brain cortex: [GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074), [SRP183521](https://www.ncbi.nlm.nih.gov/sra?term=SRP183521)

Human PBMC: [data](https://github.com/SydneyBioX/scJoint/blob/main/data.zip), [Fragment files](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156478)


### Run DGTA

1. Run DGTA/graph_build/main_graph.R

2. Then, you can run DGTA in batch_main.ipynb

### input 

Seeing DGTA/data/human40k/count.list.rds and label.list.rds

The count.list contains scRNA-seq counts and scATAC-seq ACTIVIYT counts, with the same gene dimension

The label.list contains cell type of scRNA-seq and scATAC-seq
