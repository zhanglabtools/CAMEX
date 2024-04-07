# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/11/14 23:07
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : kBET.py
# @IDE     : PyCharm

# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/11/14 22:39
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : script_kbet.py
# @IDE     : PyCharm
import numpy as np
import scanpy as sc
import psutil
import pandas as pd
import anndata as ad
from typing import Tuple, Union
from scipy.stats import chi2


def calculate_kbet(
        adata: ad.AnnData,
        use_rep: str = "delta",
        batch_col: str = "batch_indices",
        n_neighbors: int = 25,
        alpha: float = 0.05,
        random_state: int = 0,
        n_jobs: Union[None, int] = None,
        calc_knn: bool = True
) -> Tuple[float, float, float]:
    """Calculates the kBET metric of the data.

    kBET measures if cells from different batches mix well in their local
    neighborhood.

    Args:
        adata: annotated data matrix.
        use_rep: the embedding to be used. Must exist in adata.obsm.
        batch_col: a key in adata.obs to the batch column.
        n_neighbors: # nearest neighbors.
        alpha: acceptance rate threshold. A cell is accepted if its kBET
            p-value is greater than or equal to alpha.
        random_state: random seed. Used only if method is "hnsw".
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        calc_knn: whether to re-calculate the kNN graph or reuse the one stored
            in adata.

    Returns:
        stat_mean: mean kBET chi-square statistic over all cells.
        pvalue_mean: mean kBET p-value over all cells.
        accept_rate: kBET Acceptance rate of the sample.
    """

    # _logger.info('Calculating kbet...')
    assert batch_col in adata.obs
    if adata.obs[batch_col].dtype.name != "category":
        # _logger.warning(f'Making the column {batch_col} of adata.obs categorical.')
        adata.obs[batch_col] = adata.obs[batch_col].astype('category')

    ideal_dist = (
        adata.obs[batch_col].value_counts(normalize=True, sort=False).values
    )  # ideal no batch effect distribution
    nsample = adata.shape[0]
    nbatch = ideal_dist.size

    attr_values = adata.obs[batch_col].values.copy()
    attr_values.categories = range(nbatch)
    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, calc_knn)

    # partition into chunks
    n_jobs = min(_eff_n_jobs(n_jobs), nsample)
    starts = np.zeros(n_jobs + 1, dtype=int)
    quotient = nsample // n_jobs
    remainder = nsample % n_jobs
    for i in range(n_jobs):
        starts[i + 1] = starts[i] + quotient + (1 if i < remainder else 0)

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", n_jobs=n_jobs):
        kBET_arr = np.concatenate(
            Parallel()(
                delayed(_calculate_kbet_for_one_chunk)(
                    knn_indices[starts[i]: starts[i + 1], :], attr_values, ideal_dist, n_neighbors
                )
                for i in range(n_jobs)
            )
        )

    res = kBET_arr.mean(axis=0)
    stat_mean = res[0]
    pvalue_mean = res[1]
    accept_rate = (kBET_arr[:, 1] >= alpha).sum() / nsample  # this is kBET

    return (stat_mean, pvalue_mean, accept_rate)


def _calculate_kbet_for_one_chunk(knn_indices, attr_values, ideal_dist, n_neighbors):
    dof = ideal_dist.size - 1

    ns = knn_indices.shape[0]
    results = np.zeros((ns, 2))
    for i in range(ns):
        # NOTE: Do not use np.unique. Some of the batches may not be present in
        # the neighborhood.
        observed_counts = pd.Series(attr_values[knn_indices[i, :]]).value_counts(sort=False).values
        expected_counts = ideal_dist * n_neighbors
        stat = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


def _eff_n_jobs(n_jobs: Union[None, int]) -> int:
    """If n_jobs <= 0, set it as the number of physical cores _cpu_count"""
    if n_jobs is None:
        return 1
    return int(n_jobs) if n_jobs > 0 else _cpu_count


_cpu_count: Union[None, int] = psutil.cpu_count(logical=False)


def _get_knn_indices(adata: ad.AnnData,
                     use_rep: str = "delta",
                     n_neighbors: int = 25,
                     random_state: int = 0,
                     calc_knn: bool = True
                     ) -> np.ndarray:
    if calc_knn:
        assert use_rep == 'X' or use_rep in adata.obsm, f'{use_rep} not in adata.obsm and is not "X"'
        neighbors = sc.Neighbors(adata)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, use_rep=use_rep, random_state=random_state,
                                    write_knn_indices=True)
        adata.obsp['distances'] = neighbors.distances
        adata.obsp['connectivities'] = neighbors.connectivities
        adata.obsm['knn_indices'] = neighbors.knn_indices
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'knn_indices_key': 'knn_indices',
            'params': {
                'n_neighbors': n_neighbors,
                'use_rep': use_rep,
                'metric': 'euclidean',
                'method': 'umap'
            }
        }
    else:
        assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
        assert adata.uns['neighbors']['params'][
                   'n_neighbors'] >= n_neighbors, f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"

    return adata.obsm['knn_indices']


if __name__ == '__main__':
    print('hello world')
