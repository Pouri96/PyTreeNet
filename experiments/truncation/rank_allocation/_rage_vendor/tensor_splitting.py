"""The QR decomposition, and the parameters of the truncated SVD (which is PyTreeNet's)."""
from typing import Tuple
from dataclasses import dataclass
import numpy as np

from pytreenet.util.tensor_util import tensor_matricization
from pytreenet.util.ttn_exceptions import positivity_check
# Re-exported, not redefined: upstream compares by identity (``mode is SplitMode.KEEP``).
from pytreenet.util.tensor_splitting import SplitMode


__all__ = ["SVDParameters", "SplitMode", "tensor_qr_decomposition"]


def tensor_qr_decomposition(tensor: np.ndarray,
                            q_legs: Tuple[int,...],
                            r_legs: Tuple[int,...],
                            mode: SplitMode = SplitMode.REDUCED) -> Tuple[np.ndarray,np.ndarray]:
    """
    Computes the QR decomposition of a tensor with respect to the given legs.

    Args:
        tensor: Tensor on which the QR-decomp is to applied.
        q_legs: Legs of tensor that should be associated to the
            Q tensor after QR-decomposition.
        r_legs: Legs of tensor that should be associated to the
            R tensor after QR-decomposition.
        mode: Reduced returns a QR deocomposition with
            minimum dimension between Q and R. Full returns the decomposition
            with dimension between Q and R as the output dimension of Q. Keep
            causes Q to have the same shape as the input tensor. Defaults to
            SplitMode.REDUCED.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (Q, R)::

                  |2                             |1
                __|_      r_legs = (1, )       __|_        ____
               |    |     q_legs = (0,2)      |    |      | R  |
            ___|    |___  -------------->  ___| Q  |______|____|____
            0  |____|  1                   0  |____| 2   0        1

    """
    correctly_order = q_legs + r_legs == list(range(len(q_legs) + len(r_legs)))
    matrix = tensor_matricization(tensor, q_legs, r_legs,
                                  correctly_ordered=correctly_order)
    q, r = np.linalg.qr(matrix, mode=mode.numpy_qr_mode())
    shape = tensor.shape
    q_shape = _determine_tensor_shape(shape, q, q_legs, output=True)
    r_shape = _determine_tensor_shape(shape, r, r_legs, output=False)
    q = q.reshape(q_shape)
    r = r.reshape(r_shape)
    if mode is SplitMode.KEEP:
        orig_bond_dim = np.prod(r.shape[1:])
        diff = orig_bond_dim - q.shape[-1]
        padding_q = [(0,0)] * (q.ndim-1)
        padding_q.append((0,diff))
        q = np.pad(q,padding_q)
        padding_r = [(0,diff)]
        padding_r.extend([(0,0)]*(r.ndim-1))
        r = np.pad(r,padding_r)
    return q, r

def _determine_tensor_shape(old_shape: Tuple[int,...],
                            matrix: np.ndarray,
                            legs: Tuple[int,...],
                            output: bool = True) -> Tuple[int,...]:
    """
    Determines the shape of a tensor after a decomposition.

    Determines the new shape a matrix is to be reshaped to after a decomposition
    of a tensor with some original shape, to again obtain a tensor.
    Works only if all legs to be reshaped are combined in the input or output
    leg of the matrix.

    Args:
        old_shape: Shape of the original tensor.
        matrix: Matrix to be reshaped.
        legs: Which legs of the original tensor are associated to
            the matrix.
        output: If the legs of the original tensor are
            associated to the input or output of matrix. Defaults to True.

    Returns:
        Tuple[int]: New shape to which matrix is to be reshaped.
    """
    leg_shape = [old_shape[i] for i in legs]
    if output:
        matrix_dimension = [matrix.shape[1]]
        leg_shape.extend(matrix_dimension)
        new_shape = leg_shape
    else:
        matrix_dimension = [matrix.shape[0]]
        matrix_dimension.extend(leg_shape)
        new_shape = matrix_dimension

    return tuple(new_shape)

@dataclass
class SVDParameters:
    r"""
    Holds all the parameters required for a truncated singular value
    decomposition.

    Attributes:
        max_bond_dim: Cap on every bond's rank. For a cap on total size, see
            :class:`~rage.util.bug_util.ParameterBudgetConfig`. Defaults to infinity.
        rel_tol: singular values s for which
            (s / largest singular value) < rel_tol are truncated. Defaults to
            1e-12.
        total_tol: singular values s for which s < total_tol
            are truncated. Defaults to 1e-12.
        random: Use PyTreeNet's randomized SVD instead of the exact one. Defaults to False.
    """
    max_bond_dim: int = float("inf")
    rel_tol: float = 1e-12
    total_tol: float = 1e-12
    random: bool = False

    def __post_init__(self):
        """
        Check the validity of the parameters.
        """
        self.check_truncation_parameters()

    def check_truncation_parameters(self):
        """
        Checks if the truncation parameters are valid.

        The maximum bond dimension has to be a positive integer or infinity.
        The relative tolerance has to be in [0, 1) or -infinity.
        The total tolerance has to be positive or -infinity.
        """
        max_bond_dim = self.max_bond_dim
        if (not isinstance(max_bond_dim,int)) and (max_bond_dim != float("inf")):
            raise TypeError(f"'max_bond_dim' has to be int not {type(max_bond_dim)}!")
        positivity_check(max_bond_dim, "max_bond_dim")
        rel_tol = self.rel_tol
        if (rel_tol < 0) and (rel_tol != float("-inf")):
            raise ValueError("'rel_tol' has to be positive or -inf.")
        if rel_tol >= 1:
            # Usually a bond dimension passed by position.
            raise ValueError(
                f"'rel_tol' is a fraction of the largest singular value and has to be "
                f"below 1, not {rel_tol}. A value this large truncates to rank 1; if a "
                "bond dimension was meant, pass max_bond_dim.")
        total_tol = self.total_tol
        if (total_tol < 0) and (total_tol != float("-inf")):
            raise ValueError("'total_tol' has to be positive or -inf.")

