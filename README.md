# CAMEX

## Overview
Single-cell RNA-seq (scRNA-seq) data from multiple species present remarkable opportunities 
to explore cellular origins and evolution. However, integrating and annotating scRNA-seq data 
across different species remains challenging due to the variations in sequencing techniques, 
ambiguity of homologous relationships, and limited biological knowledge. To tackle above 
challenges, we introduce CAMEX, a heterogeneous Graph Neural Network (GNN) tool which 
leverages many-to-many homologous relationships for integration, alignment and annotation 
of scRNA-seq data from multiple species. Notably, CAMEX outperforms state-of-the-art (SOTA) 
methods in terms of integration on various cross-species benchmarking datasets (ranging from 
one to eleven species). Besides, CAMEX facilitates the alignment of diverse species across 
different developmental stages, significantly enhancing our understanding of organ and 
organism origins. Furthermore, CAMEX makes it easier to detect species-specific cell types 
and marker genes through cell and gene embedding. In short, CAMEX holds the potential to 
provide invaluable insights into how evolutionary forces operate across different species 
at the single cell resolution. 

![](./CAMEX_overview.png)

## Doc
The latest doc can be found [here](https://camex.readthedocs.io/en/latest/index.html).

## Prerequisites

### Data

We collected several cross-species datasets and corresponding many-to-many homologous genes which can be 
regarded as the benchmark to evaluate CAMEX with other baseline methods, and can be downloaded from
[here](https://drive.google.com/drive/folders/1rwdjEvWFEFw82a0x2JzMi2jXICbUc5eb?usp=sharing).

Put the downloaded dataset into each file as follows:

|- analysis

|-- 1liver

|--- dataset

|---- *.h5ad, many-to-many homologous genes.csv

In addition, If you want to use the many-to-many homologous genes of your own species, 
you can download them from the ensembl website [here](https://asia.ensembl.org/index.html).

### Environment

It is recommended to use a Python version  `3.9`.
* set up conda environment for CAMEX:
```
conda create -n CAMEX python==3.9
```
* install CAMEX:
```
conda activate CAMEX
```

* You need to choose the appropriate dependency pytorch and dgl for your own environment, 
and we recommend the following pytorch==1.13.1 and dgl==0.9.0 with cudatoolkit==11.6:
```
conda install cudatoolkit=11.6 -c conda-forge
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install dgl-cu116 -f https://data.dgl.ai/wheels/repo.html
```
The other versions of pytorch and dgl can be installed from
[torch](https://pytorch.org/) and [dgl](https://www.dgl.ai/pages/start.html).


## Installation
You can install CAMEX as follows:
```
git clone https://github.com/zhanglabtools/CAMEX.git
cd CAMEX
python setup.py bdist_wheel sdist
cd dist
pip install CAMEX-0.0.2.tar.gz
```

## Tutorials
The following are detailed tutorials. All tutorials were carried out on a notebook with a 11800H cpu and a 3070 8G gpu.

1. [CAMEX achieves competitive integration performance in a cross-species scenarios](./analysis/1liver/Integrate_liver_across_4_species.ipynb).

2. [CAMEX uncovers the conserved differentiation process in the testis across 11 species](./analysis/2testis/Integrate_testis_across_11_species.ipynb).

3. [CAMEX aligns various development stages of seven organs across seven different species](./analysis/3bulk/Integrate_RNAseq_across_11_species.ipynb).

4. [CAMEX could achieve more accurate integration and annotation performance in both relatives and distant species](./analysis/4cortex_annotation/integration_annotation_in_relatives_distant_species.ipynb).

5. [CAMEX facilitates the discovery of new populations and markers in Primate dlPFC](./analysis/5micro_mapping/discovery_new_populations_markers.ipynb).


## Params details
We created a params_template to explain what each of these parameters does. 
[params_template](./params_template.py)

For a more detailed description of params and train CAMEX on your own dataset, 
please refer to: [params_descriptions](./analysis/1liver/params_descriptions.ipynb)
