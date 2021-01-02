import numpy as np
from numba import njit


@njit
def greedy_matching(cost_matrix, cost_threshold):
    num_rows, num_cols = cost_matrix.shape
    cost_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(cost_1d)
    index_2d = np.stack((index_1d // num_cols, index_1d % num_cols), axis=1)

    # init matching result
    cols_for_rows = [-1 for i in range(num_rows)]  # num_rows * [-1]
    rows_for_cols = [-1 for i in range(num_cols)]  # num_cols * [-1]

    # match
    assoc_rows, assoc_cols = [], []
    for row_id, col_id in index_2d:
        if cost_matrix[row_id, col_id] >= cost_threshold:
            break
        if cols_for_rows[row_id] == -1 and rows_for_cols[col_id] == -1:
            cols_for_rows[row_id] = col_id
            rows_for_cols[col_id] = row_id
            # log associated row & col
            assoc_rows.append(row_id)
            assoc_cols.append(col_id)

    matched_pair = list(zip(assoc_rows, assoc_cols))
    unassoc_rows = list(set([i for i in range(num_rows)]) - set(assoc_rows))
    unassoc_cols = list(set([j for j in range(num_cols)]) - set(assoc_cols))
    return matched_pair, unassoc_rows, unassoc_cols

