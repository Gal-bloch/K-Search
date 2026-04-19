"""Seed submission for the MLX Mamba selective scan forward task.

Correctness-first baseline (sequential scan over sequence length).
"""

from __future__ import annotations


def seed_code() -> str:
    return '''\
import mlx.core as mx


def _softplus(x):
    return mx.where(x <= 20.0, mx.log1p(mx.exp(x)), x)


def run(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    dtype_in = u.dtype

    u = u.astype(mx.float32)
    delta = delta.astype(mx.float32)

    if delta_bias is not None:
        delta = delta + mx.expand_dims(delta_bias.astype(mx.float32), axis=-1)

    if delta_softplus:
        delta = _softplus(delta)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    B = B.astype(mx.float32)
    C = C.astype(mx.float32)

    is_variable_B = B.ndim >= 3
    is_variable_C = C.ndim >= 3

    deltaA = mx.exp(mx.expand_dims(delta, axis=-1) * mx.reshape(A.astype(mx.float32), (1, dim, 1, dstate)))
    delta_u = delta * u

    if not is_variable_B:
        deltaB_u = mx.expand_dims(delta_u, axis=-1) * mx.reshape(B, (1, dim, 1, dstate))
    else:
        if B.ndim == 3:
            deltaB_u = mx.expand_dims(delta_u, axis=2) * mx.expand_dims(B, axis=1)
            deltaB_u = mx.transpose(deltaB_u, (0, 1, 3, 2))
        else:
            ngroups = B.shape[1]
            H = dim // ngroups
            B_expanded = mx.repeat(B, H, axis=1)
            deltaB_u = mx.expand_dims(delta_u, axis=2) * B_expanded
            deltaB_u = mx.transpose(deltaB_u, (0, 1, 3, 2))

    if is_variable_C and C.ndim == 4:
        ngroups_c = C.shape[1]
        H_c = dim // ngroups_c
        C = mx.repeat(C, H_c, axis=1)

    x = mx.zeros((batch, dim, dstate), dtype=mx.float32)
    ys = []
    for i in range(seqlen):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        if not is_variable_C:
            y = mx.sum(x * mx.reshape(C, (1, dim, dstate)), axis=-1)
        else:
            if C.ndim == 3:
                y = mx.sum(x * mx.expand_dims(C[:, :, i], axis=1), axis=-1)
            else:
                y = mx.sum(x * C[:, :, :, i], axis=-1)
        ys.append(y)

    out = mx.stack(ys, axis=2)

    if D is not None:
        out = out + u * mx.reshape(D.astype(mx.float32), (1, -1, 1))

    if z is not None:
        z_f = z.astype(mx.float32)
        out = out * (z_f * mx.sigmoid(z_f))

    out = out.astype(dtype_in)
    mx.eval(out)
    return out
'''
