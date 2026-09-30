"""Minimal, exact AGOP XAI in PyTorch.

The model may return either one scalar per input or a vector-valued forecast.  For
a vector forecast, ``target`` chooses the scalar scientific diagnostic to explain::

    factor = fit_agop(
        model,
        standardized_training_inputs,
        target=lambda forecast: forecast[:, 0],
        batch_size=256,
    )
    explanations = factor.explain(standardized_query_inputs)

This file uses every supplied reference input to form
``M = mean_i grad(f_i) grad(f_i).T`` and returns the unit vector proportional to
``M**(1/2) (x - center)``.  The default center is exactly zero, as in the paper;
pass another center explicitly if desired.  Inputs should normally be standardized
before calling this code because feature scaling changes the AGOP geometry.

The exact implementation stores a dense d-by-d matrix and diagonalizes it, so it
is intended as a clear reference implementation for moderate input dimension.
Very high-dimensional weather states require a separately chosen matrix-free or
low-rank backend, but the definition of the explanation is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

ScalarTarget = Callable[[object], Tensor]


@dataclass(frozen=True)
class AGOPFactor:
    """Exact factorization of the empirical AGOP square root."""

    basis: Tensor
    root_eigenvalues: Tensor
    center: Tensor
    input_shape: tuple[int, ...]

    def explain(self, inputs: Tensor) -> Tensor:
        """Return one unit-norm AGOP explanation per input, on the CPU."""

        if tuple(inputs.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected inputs with trailing shape {self.input_shape}, "
                f"received {tuple(inputs.shape[1:])}."
            )
        flat = inputs.detach().to(device="cpu", dtype=torch.float64).flatten(1)
        if not torch.isfinite(flat).all():
            raise ValueError("Explanation inputs must be finite.")

        projected = (flat - self.center) @ self.basis
        transformed = (projected * self.root_eigenvalues) @ self.basis.T
        norms = torch.linalg.vector_norm(transformed, dim=1)
        bad = (~torch.isfinite(norms)) | (norms <= torch.finfo(torch.float64).eps)
        if bad.any():
            rows = torch.nonzero(bad, as_tuple=False).flatten().tolist()
            raise ValueError(f"AGOP explanation is undefined at rows {rows}.")
        return (transformed / norms[:, None]).reshape(inputs.shape)


def fit_agop(
    model: nn.Module,
    reference_inputs: Tensor,
    *,
    target: ScalarTarget | None = None,
    center: Tensor | None = None,
    batch_size: int = 256,
) -> AGOPFactor:
    """Fit the exact empirical AGOP from a fixed reference population.

    ``target(model(batch))`` must return one scalar per batch member.  If
    ``target`` is omitted, the model itself must return shape ``(batch,)`` or
    ``(batch, 1)``.  The model must process batch members independently.
    """

    if reference_inputs.ndim < 2 or reference_inputs.shape[0] == 0:
        raise ValueError("reference_inputs must have shape (n, ...) with n > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not torch.isfinite(reference_inputs).all():
        raise ValueError("reference_inputs must be finite.")

    input_shape = tuple(reference_inputs.shape[1:])
    dimension = reference_inputs[0].numel()
    center_flat = (
        torch.zeros(dimension, dtype=torch.float64)
        if center is None
        else torch.as_tensor(center, dtype=torch.float64).reshape(-1)
    )
    if center_flat.numel() != dimension or not torch.isfinite(center_flat).all():
        raise ValueError(f"center must contain {dimension} finite values.")

    device, dtype = _model_device_and_dtype(model, reference_inputs)
    matrix = torch.zeros((dimension, dimension), dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, reference_inputs.shape[0], batch_size):
            batch = reference_inputs[start : start + batch_size]
            batch = batch.detach().to(device=device, dtype=dtype).requires_grad_(True)
            with torch.enable_grad():
                output = model(batch)
                scores = _scalar_scores(output, target, batch.shape[0])
                gradients = torch.autograd.grad(
                    scores,
                    batch,
                    grad_outputs=torch.ones_like(scores),
                )[0]
            rows = gradients.detach().to(device="cpu", dtype=torch.float64).flatten(1)
            if not torch.isfinite(rows).all():
                raise ValueError("The model produced non-finite input gradients.")
            matrix.addmm_(rows.T, rows)
    finally:
        model.train(was_training)

    matrix /= reference_inputs.shape[0]
    matrix = (matrix + matrix.T) / 2
    eigenvalues, basis = torch.linalg.eigh(matrix)
    tolerance = (
        64
        * torch.finfo(torch.float64).eps
        * dimension
        * max(float(eigenvalues.abs().max()), 1.0)
    )
    if float(eigenvalues.min()) < -tolerance:
        raise RuntimeError("The accumulated AGOP matrix is not positive semidefinite.")
    order = torch.arange(dimension - 1, -1, -1)
    eigenvalues = torch.where(
        eigenvalues.abs() <= tolerance,
        torch.zeros_like(eigenvalues),
        eigenvalues,
    )
    eigenvalues = eigenvalues.clamp_min(0)[order]
    basis = basis[:, order]
    return AGOPFactor(
        basis=basis,
        root_eigenvalues=eigenvalues.sqrt(),
        center=center_flat,
        input_shape=input_shape,
    )


def _scalar_scores(
    output: object,
    target: ScalarTarget | None,
    batch_size: int,
) -> Tensor:
    scores = target(output) if target is not None else output
    if not isinstance(scores, Tensor):
        raise TypeError("The model or target must return a torch.Tensor.")
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.shape != (batch_size,):
        raise ValueError(
            "Choose one scalar target per input; expected shape "
            f"({batch_size},), received {tuple(scores.shape)}."
        )
    return scores


def _model_device_and_dtype(
    model: nn.Module,
    inputs: Tensor,
) -> tuple[torch.device, torch.dtype]:
    for value in (*model.parameters(), *model.buffers()):
        if value.is_floating_point():
            return value.device, value.dtype
    dtype = inputs.dtype if inputs.is_floating_point() else torch.float32
    return inputs.device, dtype


if __name__ == "__main__":
    torch.manual_seed(7)
    example_model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    references = torch.randn(256, 4)
    queries = torch.randn(3, 4)
    example_factor = fit_agop(
        example_model,
        references,
        target=lambda forecast: forecast[:, 0] - forecast[:, 1],
    )
    print(example_factor.explain(queries))
