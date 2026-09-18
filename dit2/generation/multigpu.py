"""Multi-GPU sharded generation — spatial domain decomposition.

For large cells the VRAM bottleneck during generation is NOT the model
weights (a few M parameters) but the per-edge activations of the
equivariant conv stack: memory grows linearly with the number of edges
(≈ N × num_neighbors), and one GPU caps the cell size.  Distributing
generation over several GPUs therefore has to split the *graph*, not
just the batch.

`ShardedModel` wraps a loaded denoiser with the same ``model(data) →
displacement`` interface `generate()` already uses, but runs the
forward as P slab subgraphs on P devices:

1.  Atoms are partitioned into P contiguous slabs along the longest
    cell axis (sorted by fractional coordinate → equal atom counts, so
    compute and memory balance even for inhomogeneous structures).
2.  Each shard is grown by the model's receptive field: an L-layer
    message-passing network needs the L-hop in-neighbourhood of its
    owned atoms.  The halo is computed by L−1 reverse-BFS expansions
    over the actual edge list (dst → src), so it is exact — including
    periodic-image edges and dual-cutoff LR edges — rather than a
    geometric estimate.
3.  Each device runs a full replica of the model over its subgraph
    (weights are small; activations are the cost) and only the OWNED
    atoms' displacements are gathered back.  Because the model reads
    only ``x / edge_index / edge_attr / region / batch`` and the scalar
    conditions — never ``data.pos`` (see the G5 note in core.py) — the
    sharded forward is *numerically identical* to the full forward:
    every kept edge carries the same precomputed edge vector, and every
    scatter has ``dim_size`` pinned.

The neighbour list itself stays on the primary device (it is already
chunk-built and only ~28 bytes/edge); what moves to the other GPUs is
the dominant term — the conv-stack activations, which now scale as
``N/P + halo``.  Shards run concurrently on their own devices via
threads (CUDA kernels overlap across devices); when the device list
contains duplicates (e.g. ``cuda:0,cuda:0``) the shards run
*sequentially* instead, which trades speed for peak-memory reduction on
a single GPU.

Usage (CLI):  set ``device: cuda:0,cuda:1,cuda:2,cuda:3`` in the
generate / interface YAML (or ``--override device=cuda:0,cuda:1``).
"""

import copy
import threading

import torch
import torch.nn as nn
from torch_geometric.data import Data


def unwrap_model(m):
    """Walk the ``.model`` wrapper chain (ShardedModel → adapter → net)
    to the innermost module, where training_z / conv_type / n_regions /
    cond_names etc. live."""
    seen = set()
    while True:
        inner = getattr(m, "model", None)
        if not isinstance(inner, nn.Module) or id(inner) in seen:
            return m
        seen.add(id(m))
        m = inner


def parse_device_spec(spec):
    """Normalise a device spec into a list of device strings.

    Accepts "cuda", "cuda:0,cuda:1", "cuda:0, cuda:1", or a list.
    A bare "cuda" with no index stays a single device (the current
    behaviour); multi-GPU is opt-in via an explicit comma list.
    """
    if isinstance(spec, (list, tuple)):
        return [str(d).strip() for d in spec if str(d).strip()]
    return [d.strip() for d in str(spec).split(",") if d.strip()]


def _detect_num_convs(model):
    """Number of message-passing layers = receptive-field depth in hops."""
    for obj in (model, unwrap_model(model)):
        for name in ("convs", "interactions"):
            ml = getattr(obj, name, None)
            if isinstance(ml, nn.ModuleList) and len(ml) > 0:
                return len(ml)
    return None


class ShardedModel(nn.Module):
    """Drop-in ``model(data) → disp`` wrapper that shards the graph
    forward across ``devices`` (see module docstring).  Exact: outputs
    match the unsharded forward to float round-off.
    """

    def __init__(self, model, devices, num_convs=None):
        super().__init__()
        devs = [torch.device(d) for d in parse_device_spec(devices)]
        if not devs:
            raise ValueError("ShardedModel needs at least one device")
        self.model = model                      # primary replica (registered)
        object.__setattr__(self, "_inner", unwrap_model(model))
        self.devices = devs
        self.primary = devs[0]
        hops = num_convs or _detect_num_convs(model)
        if hops is None:
            raise ValueError(
                "Cannot detect the model's conv-layer count (no .convs / "
                ".interactions ModuleList); pass num_convs= explicitly — "
                "the shard halo must cover the full receptive field.")
        self.hops = int(hops)
        # Duplicate devices (e.g. cuda:0,cuda:0) → run shards one at a
        # time: same peak-activation reduction, no concurrency on one GPU.
        self.sequential = len({str(d) for d in devs}) < len(devs)

        model.to(self.primary).eval()
        self.replicas = [model]
        for d in devs[1:]:
            r = copy.deepcopy(model).to(d)
            r.eval()
            self.replicas.append(r)
        self._warned_halo = False

    # Introspection helpers (_get_training_z, _detect_dual_cutoff,
    # n_regions lookups, init_embed._cutoff, …) read attributes off
    # whatever object generate() holds.  Delegate anything this wrapper
    # doesn't define to the innermost model so those all keep working.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def to(self, *args, **kwargs):
        """No-op: replicas are pinned to their devices at construction.
        (generate()/CLI call ``model.to(dv)`` defensively — moving only
        the primary replica would silently desync the shard set.)"""
        return self

    # ── shard construction ────────────────────────────────────────────
    def _partition(self, data, P):
        """Owned-atom index chunks: P contiguous slabs along the longest
        cell axis (equal atom counts, spatially compact → small halo)."""
        cell = data.cell
        lens = cell.norm(dim=1)
        a = int(lens.argmax())
        frac = data.pos @ torch.linalg.inv(cell)
        order = frac[:, a].argsort()
        N = order.numel()
        bounds = [round(k * N / P) for k in range(P + 1)]
        return [order[bounds[k]:bounds[k + 1]] for k in range(P)]

    def _halo(self, own_idx, N, edge_sets, device):
        """(need_mask, keep_masks): nodes within `hops` reverse hops of
        the owned set, and per-edge-set masks of the edges whose TARGET
        is within hops−1 — exactly the edges that influence owned
        outputs through an L-layer message-passing stack."""
        m = torch.zeros(N, dtype=torch.bool, device=device)
        m[own_idx] = True
        for _ in range(self.hops - 1):
            for src, dst in edge_sets:
                m[src[m[dst]]] = True
        keep = [m[dst] for _, dst in edge_sets]
        need = m.clone()
        for (src, _), kp in zip(edge_sets, keep):
            need[src[kp]] = True
        return need, keep

    def _build_shard(self, data, own_idx, edge_sets, has_lr):
        N = data.x.size(0)
        dev0 = data.x.device
        need, keep = self._halo(own_idx, N, edge_sets, dev0)
        need_idx = need.nonzero(as_tuple=True)[0]
        g2l = torch.full((N,), -1, dtype=torch.long, device=dev0)
        g2l[need_idx] = torch.arange(need_idx.numel(), device=dev0)

        sub = Data()
        sub.x = data.x[need_idx]
        sub.num_graphs = 1
        sub.batch = torch.zeros(need_idx.numel(), dtype=torch.long, device=dev0)
        ei = edge_sets[0]
        sub.edge_index = torch.stack([g2l[ei[0][keep[0]]], g2l[ei[1][keep[0]]]])
        sub.edge_attr = data.edge_attr[keep[0]]
        if has_lr:
            lr = edge_sets[1]
            sub.lr_edge_index = torch.stack(
                [g2l[lr[0][keep[1]]], g2l[lr[1][keep[1]]]])
            sub.lr_edge_attr = data.lr_edge_attr[keep[1]]
        # Per-atom labels and the graph-level scalars the model reads.
        region = getattr(data, "region", None)
        if region is not None:
            sub.region = region[need_idx]
        # Graph-level scalars the model reads (data.dx is applied OUTSIDE
        # model(data) in generate(), so it never needs to travel).
        for name in ("energy", "_sigma_t"):
            v = getattr(data, name, None)
            if v is not None:
                sub[name] = v
        for name in getattr(self._inner, "cond_names", None) or []:
            v = getattr(data, name, None)
            if v is not None and name not in sub:
                sub[name] = v
        # Conditioning v2: the method id and RDF channels are per-graph
        # (shard-invariant) — copy wholesale.  Omitting them would make a
        # method/RDF-conditioned shard silently fall back to defaults and
        # break bit-exactness with the unsharded forward.
        for name in ("method", "temperature", "delta_e_min",
                     "energy_present", "method_present",
                     "temperature_present", "delta_e_min_present",
                     "rdf_totals", "rdf_present",
                     "rdf_partial", "rdf_pair_present"):
            v = getattr(data, name, None)
            if v is not None and name not in sub:
                sub[name] = v
        return sub, need_idx, g2l

    # ── forward ───────────────────────────────────────────────────────
    def forward(self, data):
        P = len(self.devices)
        if P == 1:
            return self.replicas[0](data)

        N = data.x.size(0)
        edge_sets = [(data.edge_index[0], data.edge_index[1])]
        has_lr = getattr(data, "lr_edge_index", None) is not None
        if has_lr:
            edge_sets.append((data.lr_edge_index[0], data.lr_edge_index[1]))

        chunks = self._partition(data, P)
        shards = [self._build_shard(data, own, edge_sets, has_lr)
                  for own in chunks]

        # If the halo swallows (almost) the whole graph, sharding gives
        # no memory benefit — tell the user once instead of failing.
        if not self._warned_halo:
            biggest = max(s[1].numel() for s in shards)
            if biggest > 0.9 * N:
                import warnings
                warnings.warn(
                    f"ShardedModel: a shard needs {biggest}/{N} atoms after "
                    f"the {self.hops}-hop halo — the cell is too small (or "
                    "the receptive field too large) for sharding to reduce "
                    "memory. Results stay correct; expect no VRAM savings.",
                    RuntimeWarning, stacklevel=2)
            self._warned_halo = True

        out = torch.empty(N, 3, device=self.primary, dtype=data.edge_attr.dtype)
        results = [None] * P

        def _run(k):
            # torch.no_grad() is thread-local — generate()'s context does
            # NOT cover these worker threads; re-enter it explicitly.
            with torch.no_grad():
                sub, need_idx, g2l = shards[k]
                d = self.devices[k]
                sub_d = sub.to(d) if d != self.primary else sub
                disp_local = self.replicas[k](sub_d)
                sel = g2l[chunks[k]].to(d)
                results[k] = disp_local[sel].to(self.primary)

        if self.sequential:
            for k in range(P):
                _run(k)
        else:
            threads = [threading.Thread(target=_run, args=(k,)) for k in range(P)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        for k in range(P):
            out[chunks[k]] = results[k]
        return out
