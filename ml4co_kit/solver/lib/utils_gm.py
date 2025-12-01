import numpy as np
import scipy.optimize
import scipy.special

def _check_and_init(K: np.ndarray, n1: int = None, n2: int = None, x0: np.ndarray = None):
    n1n2 = K.shape[0]
 
    if n1 is None and n2 is None:
        raise ValueError('Neither n1 or n2 is given.')
    if n1 is None:
        if n1n2 % n2 == 0:
            n1 = n1n2 / n2
        else:
            raise ValueError("The input size of K does not match with n2!")
    if n2 is None:
        if n1n2 % n1 == 0:
            n2 = n1n2 / n1
        else:
            raise ValueError("The input size of K does not match with n1!")
    if not n1 * n2 == n1n2:
        raise ValueError('the input size of K does not match with n1 * n2!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = np.zeros((n1, n2), dtype=K.dtype)
        x0[:] = 1. / (n1 * n2)
    v0 = x0.T.reshape((n1n2, 1))

    return n1, n2, n1n2, v0

def sinkhorn(
        s: np.ndarray, 
        nrows: np.ndarray = None, 
        ncols: np.ndarray = None,
        unmatchrows: np.ndarray = None, 
        unmatchcols: np.ndarray = None,
        dummy_row: bool = False, 
        max_iter: int = 10, 
        tau: float = 1.
    ) -> np.ndarray:
        """Sinkhorn Algorithm"""
        transposed = False
        if s.shape[0] > s.shape[1]:
            s = s.T
            transposed = True
            nrows, ncols = ncols, nrows
            unmatchrows, unmatchcols = unmatchcols, unmatchrows

        nrows = s.shape[0] if nrows is None else nrows
        ncols = s.shape[1] if ncols is None else ncols

        # operations are performed on log_s
        log_s = s / tau
        if unmatchrows is not None:
            unmatchrows = unmatchrows / tau
        if unmatchcols is not None:
            unmatchcols = unmatchcols / tau

        if dummy_row and nrows < ncols:
            dummy_count = ncols - nrows
            log_s = np.vstack([log_s, np.full((dummy_count, ncols), -100.)])
            if unmatchrows is not None:
                unmatchrows = np.concatenate([unmatchrows, np.full(dummy_count, -100.)])
            nrows = ncols

        # assign the unmatch weights
        if unmatchrows is not None and unmatchcols is not None:
            log_s = np.pad(log_s, ((0,1),(0,1)), constant_values=-np.inf)
            log_s[:nrows, ncols] = unmatchrows[:nrows]
            log_s[nrows, :ncols] = unmatchcols[:ncols]
            nrows += 1
            ncols += 1

        row_mask = np.zeros((log_s.shape[0],1), dtype=bool)
        col_mask = np.zeros((1,log_s.shape[1]), dtype=bool)
        row_mask[:nrows,0] = True
        col_mask[0,:ncols] = True

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = scipy.special.logsumexp(log_s, axis=1, keepdims=True)
                log_s = log_s - log_sum * row_mask
            else:
                log_sum = scipy.special.logsumexp(log_s, axis=0, keepdims=True)
                log_s = log_s - log_sum * col_mask

        if unmatchrows is not None and unmatchcols is not None:
            log_s = log_s[:-1,:-1]

        if dummy_row and 'dummy_count' in locals() and dummy_count > 0:
            log_s = log_s[:-dummy_count, :]

        if transposed:
            log_s = log_s.T

        return np.exp(log_s)

def hungarian(s: np.ndarray, n1=None, n2=None, unmatch1=None, unmatch2=None):
        """
        Hungarian kernel function by calling the linear sum assignment solver from Scipy.
        """
        s = -s
        if unmatch1 is not None:
            unmatch1 = -unmatch1
        if unmatch2 is not None:
            unmatch2 = -unmatch2
            
        if n1 is None:
            n1 = s.shape[0]
        if n2 is None:
            n2 = s.shape[1]
        if unmatch1 is not None and unmatch2 is not None:
            upper_left = s[:n1, :n2]
            upper_right = np.full((n1, n1), float('inf'))
            np.fill_diagonal(upper_right, unmatch1[:n1])
            lower_left = np.full((n2, n2), float('inf'))
            np.fill_diagonal(lower_left, unmatch2[:n2])
            lower_right = np.zeros((n2, n1))

            large_cost_mat = np.concatenate((np.concatenate((upper_left, upper_right), axis=1),
                                         np.concatenate((lower_left, lower_right), axis=1)), axis=0)

            row, col = scipy.optimize.linear_sum_assignment(large_cost_mat)
            valid_idx = np.logical_and(row < n1, col < n2)
            row = row[valid_idx]
            col = col[valid_idx]
        else:
            row, col = scipy.optimize.linear_sum_assignment(s[:n1, :n2])
        perm_mat = np.zeros_like(s)
        perm_mat[row, col] = 1
  
        return perm_mat

