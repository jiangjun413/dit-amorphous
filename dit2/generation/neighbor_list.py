import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list

from dit2.constants import _BB, _CL
from dit2.generation.structure import is_ortho


def _wrap_torch(pos, cell):
    """Wrap positions into cell using fractional coordinates. Any cell shape."""
    cell_inv = torch.linalg.inv(cell)
    frac = pos @ cell_inv
    frac -= frac.floor()
    return frac @ cell

def _wrap_numpy(pos, cell):
    """Wrap positions into cell using fractional coordinates. Any cell shape."""
    cell_inv = np.linalg.inv(cell)
    frac = pos @ cell_inv
    frac -= np.floor(frac)
    return frac @ cell

def _min_image_torch(d, cell, cell_inv):
    """Minimum-image displacement vectors for arbitrary cells (torch)."""
    frac = d @ cell_inv
    frac -= torch.round(frac)
    return frac @ cell

def _min_image_numpy(d, cell, cell_inv):
    """Minimum-image displacement vectors for arbitrary cells (numpy)."""
    frac = d @ cell_inv
    frac -= np.round(frac)
    return frac @ cell

###############################################################################
# NEIGHBOR LIST
###############################################################################
# --- Orthogonal-only helpers (kept for fast GPU path) ---
def _wg(p, L): f = p/L; return (f-f.floor())*L

def _wn(p, L): f = p/L; return (f-np.floor(f))*L

def _nl_cl(pw, L, n, cut, dv):
    nc = (L/cut).long().clamp(min=1); cs = L/nc.float()
    c3 = (pw/cs).long(); c3[:,0].clamp_(0,nc[0]-1); c3[:,1].clamp_(0,nc[1]-1); c3[:,2].clamp_(0,nc[2]-1)
    cl = c3[:,0]*nc[1]*nc[2]+c3[:,1]*nc[2]+c3[:,2]; od = cl.argsort(); sp = pw[od]; sc = cl[od]
    ntc = int(nc.prod().item()); cnt = torch.bincount(sc, minlength=ntc)
    off = torch.zeros(ntc+1, dtype=torch.long, device=dv); off[1:] = cnt.cumsum(0)
    a3 = torch.arange(-1,2,device=dv); nof = torch.cartesian_prod(a3,a3,a3)
    gx,gy,gz = torch.meshgrid(torch.arange(nc[0],device=dv),torch.arange(nc[1],device=dv),torch.arange(nc[2],device=dv),indexing="ij")
    ac = torch.stack([gx.reshape(-1),gy.reshape(-1),gz.reshape(-1)],1)
    nb = (ac.unsqueeze(1)+nof.unsqueeze(0))%nc
    nbl = nb[...,0]*nc[1]*nc[2]+nb[...,1]*nc[2]+nb[...,2]; del ac,nb,gx,gy,gz
    CH = max(32, min(512, 500_000//max(1, n//max(ntc,1))))
    ei,ej,ev = [],[],[]
    for s in range(0,n,CH):
        e = min(s+CH,n); co = od[s:e]; cp = sp[s:e]; cn = nbl[sc[s:e]]
        for k in range(27):
            ck = cn[:,k]; sk = off[ck]; ek = off[ck+1]; mo = int((ek-sk).max().item())
            if mo == 0: continue
            am = torch.arange(mo,device=dv); ga = (sk.unsqueeze(1)+am.unsqueeze(0)).clamp(max=n-1)
            va = am.unsqueeze(0) < (ek-sk).unsqueeze(1); np_ = sp[ga]; no_ = od[ga]
            df = np_-cp.unsqueeze(1); df -= torch.round(df/L)*L; dt = df.norm(dim=2)
            # Keep self-image edges (i == j with shift != 0) for small
            # cells where any axis L < 2·cut — those carry real periodic
            # neighbor contributions the model needs.  The dt > 1e-8
            # clause already filters out the trivial self-loop.
            mk = va & (dt<=cut) & (dt>1e-8)
            ci,oj = mk.nonzero(as_tuple=True)
            if ci.numel(): ei.append(co[ci]); ej.append(no_[ci,oj]); ev.append(df[ci,oj])
            del np_,no_,df,dt,mk,va
    del sp,sc,od,nbl,off,cnt
    if ei: return torch.stack([torch.cat(ei),torch.cat(ej)]),torch.cat(ev)
    return torch.zeros(2,0,dtype=torch.long,device=dv),torch.zeros(0,3,dtype=pw.dtype,device=dv)

def _nl_br(pw, L, n, cut, dv):
    nr = torch.ceil(cut/L).long()
    sh = torch.cartesian_prod(*[torch.arange(-r.item(),r.item()+1,device=dv,dtype=pw.dtype) for r in nr])*L
    pr = (pw.unsqueeze(0)+sh.unsqueeze(1)).reshape(-1,3); oj = torch.arange(n,device=dv).repeat(sh.shape[0])
    ch = max(1, min(n, _BB//max(1, n*sh.shape[0]*16)))
    ei,ej,ev = [],[],[]
    for s in range(0,n,ch):
        e = min(s+ch,n); cp = pw[s:e]; ci = torch.arange(s,e,device=dv)
        df = pr.unsqueeze(0)-cp.unsqueeze(1); dt = df.norm(dim=2)
        # Keep self-image edges (i == j with shift != 0).  The dt > 1e-8
        # clause already filters out the trivial self-loop.  Dropping
        # the (ci != oj) mask matches the GPU-triclinic and CPU paths
        # which both retain self-images.
        mk = (dt<=cut)&(dt>1e-8)
        ri,rj = mk.nonzero(as_tuple=True)
        if ri.numel(): ei.append(ci[ri]); ej.append(oj[rj]); ev.append(df[ri,rj])
        del df,dt,mk
    del pr,oj
    if ei: return torch.stack([torch.cat(ei),torch.cat(ej)]),torch.cat(ev)
    return torch.zeros(2,0,dtype=torch.long,device=dv),torch.zeros(0,3,dtype=pw.dtype,device=dv)

def nl_gpu_ortho(pos, cell, cut, dv):
    """GPU neighbor list — ORTHOGONAL cells only.

    The cell-list kernel (`_nl_cl`) requires at least THREE cells per axis:
    with nc == 2 the ±1 stencil offsets alias onto the SAME neighbor cell
    modulo nc ((a-1) % 2 == (a+1) % 2), so every aliased cell is visited
    multiple times and its pairs are emitted as DUPLICATE edges — up to 8×
    for a cell diagonal.  Duplicated edges double-count messages in the
    conv aggregation and silently corrupt the model input (observed: 576-
    atom GeO₂ in a 20.85 Å box at cutoff 7 Å produced 82 960 edges instead
    of 52 946 and near-random structures, while 300-atom runs — below the
    _CL threshold, brute-force path — were fine).  L >= 3*cut guarantees
    nc = floor(L/cut) >= 3 on every axis, where the 27-stencil visits 27
    distinct cells and the kernel is exact.
    """
    L = cell.diag(); pw = _wg(pos, L); n = pw.shape[0]
    ei, ea = (_nl_cl(pw, L, n, cut, dv) if (L >= 3*cut).all().item() and n >= _CL
              else _nl_br(pw, L, n, cut, dv))
    return ei, ea, pw

def nl_cpu(pn, cn, pbc, nums, cut, dv):
    """CPU neighbor list via ASE — handles ANY cell shape."""
    ii, jj, D = primitive_neighbor_list("ijD", cutoff=cut, pbc=pbc, cell=cn,
                                        positions=pn, numbers=nums)
    return (torch.from_numpy(np.stack((ii, jj))).long().to(dv),
            torch.from_numpy(D).float().to(dv))

def nl_gpu_general(pos, cell, cut, dv):
    """GPU neighbor list for general (triclinic) cells using a ghost-atom grid."""
    n = pos.shape[0]
    cell_inv = torch.linalg.inv(cell)

    # 1. Determine maximum periodic boundary shifts needed
    vol = torch.abs(torch.linalg.det(cell))
    widths = torch.zeros(3, device=dv)
    for i in range(3):
        j, k = (i+1)%3, (i+2)%3
        cross = torch.linalg.cross(cell[j], cell[k])
        widths[i] = vol / torch.linalg.norm(cross)

    reps = torch.ceil(cut / widths).long()

    # 2. Create grid of shifts
    r0 = torch.arange(-reps[0], reps[0] + 1, device=dv)
    r1 = torch.arange(-reps[1], reps[1] + 1, device=dv)
    r2 = torch.arange(-reps[2], reps[2] + 1, device=dv)
    shifts_frac = torch.cartesian_prod(r0, r1, r2).float()
    M = shifts_frac.shape[0]

    # 3. Minimum image wrap positions to [0, 1)
    frac = pos @ cell_inv
    frac -= frac.floor()
    wrapped_pos = frac @ cell

    # 4. Create ghost atoms (shape: N*M, 3).
    #
    # BUILD GHOSTS AS wrapped_pos + CARTESIAN SHIFT, never as
    # (frac + shift) @ cell.  The latter runs a SECOND, differently-shaped
    # matmul than the one that produced `wrapped_pos`, so the zero-shift
    # ghost came out only approximately equal to its own centre.  During
    # generation the incoming positions sit hundreds of A outside the cell
    # (EDM's first sigmas are ~1e2), so `frac` reaches ~1e2-1e3 and the
    # `frac -= frac.floor()` cancellation loses most of float32's mantissa:
    # the zero-shift residue measured ~1e-3 A, a thousand times the 1e-6
    # self-loop threshold below.  Every atom therefore kept a spurious
    # i==i edge of length ~1e-3 A -- exactly N of them, every step -- and a
    # near-zero edge vector is poison for an equivariant conv (degenerate
    # spherical harmonics, radial basis far below anything in training).
    # Result was a silent, total collapse of every non-orthogonal-cell run
    # (dmin ~0.05 A vs 1.9 A), since `is_ortho` is what selects this path.
    # Adding the shift in CARTESIAN space makes the zero-shift ghost
    # bitwise identical to `wrapped_pos`, which is also what the orthogonal
    # path (`_nl_br`: `pr = pw + sh`) has always done.
    shifts_cart = shifts_frac @ cell                       # (M, 3)
    ghost_pos = (wrapped_pos.unsqueeze(1)
                 + shifts_cart.unsqueeze(0)).view(-1, 3)
    # Index of the (0,0,0) shift, so the trivial self-loop can be dropped
    # EXACTLY rather than by a distance epsilon that a residue can defeat.
    zero_shift = int((shifts_frac.abs().sum(dim=1) == 0)
                     .nonzero(as_tuple=True)[0].item())

    # 5. Batched brute-force distances to prevent VRAM exhaustion
    ei_list, ej_list, ev_list = [], [], []
    chunk = max(1, 200_000_000 // (n * M))

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # diff points from center (wrapped_pos) to neighbor (ghost_pos)
        diff = ghost_pos.unsqueeze(0) - wrapped_pos[s:e].unsqueeze(1)
        dist = diff.norm(dim=2)

        mask = (dist > 1e-6) & (dist <= cut)
        ri, rj_ghost = mask.nonzero(as_tuple=True)

        if ri.numel() > 0:
            real_i = ri + s
            real_j = rj_ghost // M  # Map ghost index back to original atom index

            # Drop ONLY the trivial self-loop: same atom, zero shift.  This is
            # an exact index test, not a distance test -- the previous
            # `(real_i != real_j) | (dist > 1e-6)` was a tautology (dist > 1e-6
            # already held from `mask`), so it removed nothing, and the 1e-6
            # threshold in `mask` could not catch a ~1e-3 A residue.  Genuine
            # self-IMAGE edges (i == j, shift != 0) are kept, matching the
            # orthogonal and CPU/ASE paths.
            valid = ~((real_j == real_i) & ((rj_ghost % M) == zero_shift))
            if valid.any():
                ei_list.append(real_i[valid])
                ej_list.append(real_j[valid])
                ev_list.append(diff[ri, rj_ghost][valid])

    if ei_list:
        ei = torch.stack([torch.cat(ei_list), torch.cat(ej_list)])
        ea = torch.cat(ev_list)
        return ei, ea, wrapped_pos

    return torch.zeros(2, 0, dtype=torch.long, device=dv), torch.zeros(0, 3, device=dv), wrapped_pos

def build_nl(pos, cell, pbc, numbers, cut, dv, ortho=None):
    """Unified neighbor list builder.
    Auto-selects GPU (ortho or triclinic) or CPU path.

    Returns (edge_index, edge_attr, wrapped_pos).
    """
    if ortho is None:
        ortho = is_ortho(cell)

    if dv.type in ("cuda", "mps"):
        if ortho:
            ei, ea, pw = nl_gpu_ortho(pos, cell, cut, dv)
        else:
            ei, ea, pw = nl_gpu_general(pos, cell, cut, dv)
        return ei, ea, pw

    # CPU/ASE path — works for any cell geometry
    pn = pos.detach().cpu().numpy()
    cn = cell.detach().cpu().numpy() if isinstance(cell, torch.Tensor) else np.asarray(cell)
    # Ensure pbc is numpy-compatible
    pbc_np = np.asarray(pbc) if not isinstance(pbc, (bool, np.bool_)) else pbc
    pw = _wrap_numpy(pn, cn)
    # numbers must be numpy for ASE
    nums_np = numbers if isinstance(numbers, np.ndarray) else np.asarray(numbers)
    ei, ea = nl_cpu(pw, cn, pbc_np, nums_np, cut, dv)
    pw_t = torch.from_numpy(pw).float().to(dv)
    return ei, ea, pw_t

###############################################################################
# VERLET LIST  (cell-geometry-aware)
###############################################################################
class VL:
    _MB = 2<<30
    def __init__(self, cut, skin, dv, na=0):
        self.cut = cut; self.dv = dv; self.gpu = dv.type in ("cuda", "mps")
        if na > 0 and skin > 0:
            est = int(na*50*((cut+skin)/cut)**3)*28
            if est > self._MB:
                mr = cut*((self._MB/28/max(na*50,1))**(1/3)); skin = max(0, mr-cut)
                if skin < .05: skin = 0
        self.skin = skin; self.ceff = cut+skin
        self._ref = None; self._ei = None; self._ea = None
        self._nst = 0; self.nr = 0; self.nu = 0
        self._ortho = None; self._cell_inv = None

    def _detect_ortho(self, cell):
        """Cache orthogonality check and cell inverse."""
        self._ortho = is_ortho(cell)
        self._cell_inv = torch.linalg.inv(cell)

    def _wrap(self, data):
        """Wrap positions into cell. Works for any cell shape."""
        cell = data.cell
        if self._ortho is None:
            self._detect_ortho(cell)
        if self._ortho:
            L = cell.diag()
            if self.gpu:
                pw = _wg(data.pos, L)
            else:
                pw = torch.from_numpy(
                    _wn(data.pos.detach().cpu().numpy(), L.detach().cpu().numpy())
                ).float().to(self.dv)
        else:
            # Use cached inverse for efficiency
            frac = data.pos @ self._cell_inv
            frac -= frac.floor()
            pw = frac @ cell
        data.pos = pw
        return pw

    def _min_image(self, d, cell):
        """Minimum-image displacement for stored cell."""
        if self._ortho:
            L = cell.diag()
            d -= torch.round(d / L) * L
            return d
        else:
            return _min_image_torch(d, cell, self._cell_inv)

    def _filt(self, ei, ea):
        d = ea.norm(dim=1); k = d <= self.cut; del d
        if k.all(): del k; return ei, ea
        r = ei[:,k], ea[k]; del k; return r

    def update(self, data):
        cell = data.cell
        if self._ortho is None:
            self._detect_ortho(cell)

        pw = self._wrap(data)
        reb = self._ref is None or self._ei is None or self.skin <= 0
        if not reb:
            d = pw - self._ref
            d = self._min_image(d, cell)
            # M7: trigger rebuild at max_displacement > skin*0.49 (not 0.5).
            # The textbook trigger is exactly half the skin (two atoms each
            # moving skin/2 toward each other closes the buffer entirely),
            # but float round-off can leave a pair right at the cutoff in
            # the "no rebuild needed" branch — and then the filter `d <=
            # cut` evaluates the pair as missing.  Triggering ~1% earlier
            # gives a safety margin without measurably increasing rebuild
            # frequency.
            reb = d.norm(dim=1).max().item() > self.skin * 0.49
            del d

        if reb:
            ei, ea, pw2 = build_nl(pw, cell, data.pbc, data.numbers,
                                   self.ceff, self.dv, ortho=self._ortho)
            data.pos = pw2
            if self.skin > 0:
                # Store the per-image edge VECTORS, not just the index pairs.
                # Reuse steps update them with disp[j] - disp[i]; recomputing
                # pw[j] - pw[i] + min-image instead would collapse all images
                # of a pair onto one vector and zero the self-image edges
                # whenever any cell axis < 2*ceff.
                self._ei = ei; self._ea = ea; self._nst = ei.shape[1]
                self._ref = data.pos.clone()
                data.edge_index, data.edge_attr = self._filt(ei, ea)
            else:
                data.edge_index = ei; data.edge_attr = ea
            self.nr += 1
        else:
            ei = self._ei; ne = ei.shape[1]; CH = 2_000_000
            # Per-atom displacement since the rebuild reference.  Each atom
            # has moved < 0.49*skin (the rebuild trigger), so min_image
            # recovers the true displacement even when the wrapped
            # coordinate jumped across a cell boundary.  Delta-updating the
            # stored per-image vectors keeps multi-image and self-image
            # edges exact — the same trick RattleParticles uses train-side.
            disp = self._min_image(pw - self._ref, cell)
            if ne <= CH:
                df = self._ea + disp[ei[1]] - disp[ei[0]]
                data.edge_index, data.edge_attr = self._filt(ei, df)
            else:
                ki, ae = [], []
                for s in range(0, ne, CH):
                    e = min(s + CH, ne)
                    df = self._ea[s:e] + disp[ei[1, s:e]] - disp[ei[0, s:e]]
                    dt = df.norm(dim=1); kc = dt <= self.cut; del dt
                    if kc.any():
                        ki.append(kc.nonzero(as_tuple=True)[0] + s)
                        ae.append(df[kc])
                    del df, kc
                if ki:
                    ka = torch.cat(ki)
                    data.edge_index = ei[:, ka]
                    data.edge_attr = torch.cat(ae)
                    del ki, ae, ka
                else:
                    data.edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.dv)
                    data.edge_attr = torch.zeros(0, 3, dtype=pw.dtype, device=self.dv)
                    del ki, ae
            del disp
            self.nu += 1

    @property
    def stats(self):
        t = self.nr+self.nu
        if t == 0: return "—"
        si = f"skin={self.skin:.2f}" if self.skin > 0 else "skin=OFF"
        ot = "ortho" if self._ortho else "general"
        return (f"{self.nr}rb/{self.nu}ru({100*self.nu/t:.0f}%,{si},{ot})" if self.nu
                else f"{self.nr}rb({si},{ot})")
