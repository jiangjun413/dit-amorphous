import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase.neighborlist import primitive_neighbor_list
from ase.visualize.plot import plot_atoms

from dit2.constants import _CT, _LT, _atom_color
from dit2.generation.structure import is_ortho
from dit2.generation.neighbor_list import _min_image_numpy

# This is a leaf module — it deliberately does NOT call
# ``matplotlib.use('Agg')`` here.  The CLI / UI entry points configure
# the backend before importing this module so headless runs render with
# Agg; interactive (Jupyter) users keep their own backend.


def comp_cn(atoms, metals, cuts, o="O"):
    if not metals: return {}
    sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.cell)
    ortho = is_ortho(cell)

    if len(sym) < _CT and ortho:
        # Fast path — orthogonal cells only
        pos, L = atoms.get_positions(), atoms.cell.lengths(); op = pos[sym==o]; r = {}
        for el in metals:
            mp = pos[sym==el]
            if not mp.shape[0] or not op.shape[0]: continue
            d = op[np.newaxis]-mp[:,np.newaxis]; d -= np.round(d/L)*L
            r[el] = np.sum(np.linalg.norm(d,axis=2) <= cuts.get(el,3), axis=1).astype(int)
        return r
    elif len(sym) < _CT and not ortho:
        # Fast path — general cells via fractional minimum image
        pos = atoms.get_positions(); op = pos[sym==o]
        cell_inv = np.linalg.inv(cell); r = {}
        for el in metals:
            mp = pos[sym==el]
            if not mp.shape[0] or not op.shape[0]: continue
            d = op[np.newaxis] - mp[:,np.newaxis]
            d = _min_image_numpy(d.reshape(-1, 3), cell, cell_inv).reshape(d.shape)
            r[el] = np.sum(np.linalg.norm(d, axis=2) <= cuts.get(el, 3), axis=1).astype(int)
        return r

    # Large system — use ASE neighbor list (handles any cell)
    pos = atoms.get_positions()
    oi = np.where(sym==o)[0]; mc = max(cuts.get(e,3) for e in metals)
    ia,ja,da = primitive_neighbor_list("ijd",cutoff=mc,pbc=atoms.pbc,cell=cell,positions=pos,numbers=atoms.numbers)
    omj = np.isin(ja,oi); r = {}
    for el in metals:
        mi = np.where(sym==el)[0]
        if not len(mi): continue
        v = np.isin(ia,mi) & omj & (da<=cuts.get(el,3)); cn = np.zeros(len(mi),dtype=int)
        if v.any(): cn = np.bincount(ia[v],minlength=len(sym))[mi]
        r[el] = cn
    return r

def comp_rdf(atoms, rmax=8, nb=200):
    """Partial radial distribution functions g_{AB}(r).

    ASE's ``primitive_neighbor_list`` returns each unordered pair in both
    directions, so the ``ia<ja`` filter halves the entries uniformly for
    both A–A and A–B branches.  For finite systems the canonical
    normalization is

        g_{AA}(r) = c / [ N_A (N_A-1)/2 · 4πr²Δr / V ]      (self pair)
        g_{AB}(r) = c / [ N_A N_B       · 4πr²Δr / V ]      (cross pair)

    Using ``N_A(N_A-1)`` instead of ``N_A²`` on the diagonal makes
    g_{AA}(r) → 1 exactly at large r for any finite N; the legacy
    ``N_A²/2`` form gave g_{AA}(r) → (N_A-1)/N_A which differs by
    half-a-percent for typical amorphous-oxide cells (N ~ 200) and is
    visible on small training cells.
    """
    pos,sym = atoms.get_positions(),np.array(atoms.get_chemical_symbols())
    cell,vol = np.array(atoms.cell),atoms.get_volume()
    # Skip the unphysical r=0 .. 0.5 Å bins.  Physical g(r) is identically
    # zero for r below the hard-core radius (~1 Å for oxides), and ASE's
    # neighbor list anyway excludes self-loops.  The legacy `linspace(0,…)`
    # produced ghost bin centers at e.g. r=0.02 Å that confused users
    # reading the CSV output.
    rmin = 0.5
    re = np.linspace(rmin,rmax,nb+1); rm = .5*(re[:-1]+re[1:]); sh = (4/3)*np.pi*(re[1:]**3-re[:-1]**3)
    sp = sorted(set(sym))
    ia,ja,da = primitive_neighbor_list("ijd",cutoff=rmax,pbc=atoms.pbc,cell=cell,positions=pos,numbers=atoms.numbers)
    si,sj = sym[ia],sym[ja]; rdf = {}
    for xi,s1 in enumerate(sp):
        for s2 in sp[xi:]:
            mk = (si==s1)&(sj==s2)&(ia<ja) if s1==s2 else (((si==s1)&(sj==s2))|((si==s2)&(sj==s1)))&(ia<ja)
            d = da[mk]; n1,n2 = np.sum(sym==s1),np.sum(sym==s2)
            if not n1 or not n2: continue
            c,_ = np.histogram(d,bins=re)
            if s1 == s2:
                # Self-pair: N_A(N_A-1)/2 unordered pairs in the ideal gas.
                # Skip when only one atom of this species exists (no pairs).
                if n1 < 2:
                    continue
                nm = n1 * (n1 - 1) * sh / (2.0 * vol)
            else:
                nm = n1 * n2 * sh / vol
            with np.errstate(divide="ignore",invalid="ignore"): rdf[f"{s1}-{s2}"] = (rm,np.where(nm>0,c/nm,0))
    return rdf

def comp_rdf_from_pairs(sym_i, sym_j, d, counts, vol, rmax=8, nb=200):
    """Partial g_AB(r) from a PRECOMPUTED directed pair list.

    Edge-reuse twin of ``comp_rdf``: identical bins and ideal-gas
    normalization, but takes the (i→j symbol, j symbol, distance) arrays of a
    DIRECTED neighbor list instead of recomputing one.  The training graph
    builders already hold every pair distance out to the graph cutoff (which
    exceeds the RDF range), so the expensive neighbor-list step — ~1 s per
    structure on big cells, and the dominant cost of the RDF precompute —
    is eliminated; only the histogram remains (~ms).

    A directed list contains each unordered pair twice (i→j and j→i), for
    self- and cross-species alike, so every histogram is divided by 2 —
    equivalent to ``comp_rdf``'s ``ia<ja`` filter.  Distances beyond ``rmax``
    fall outside the bin range and are ignored, so a longer-cutoff edge list
    needs no pre-filtering.

    counts : dict {symbol: n_atoms}; vol : cell volume (Å³).
    Returns the same ``{"A-B": (rm, g)}`` dict as ``comp_rdf``.
    """
    rmin = 0.5
    re_ = np.linspace(rmin, rmax, nb + 1)
    rm = .5 * (re_[:-1] + re_[1:])
    sh = (4 / 3) * np.pi * (re_[1:] ** 3 - re_[:-1] ** 3)
    sym_i = np.asarray(sym_i); sym_j = np.asarray(sym_j)
    d = np.asarray(d)
    sp = sorted(counts)
    rdf = {}
    for xi, s1 in enumerate(sp):
        for s2 in sp[xi:]:
            if s1 == s2:
                mk = (sym_i == s1) & (sym_j == s1)
            else:
                mk = ((sym_i == s1) & (sym_j == s2)) | \
                     ((sym_i == s2) & (sym_j == s1))
            n1, n2 = counts[s1], counts[s2]
            if not n1 or not n2:
                continue
            c, _ = np.histogram(d[mk], bins=re_)
            c = c / 2.0            # directed list: each unordered pair twice
            if s1 == s2:
                if n1 < 2:
                    continue
                nm = n1 * (n1 - 1) * sh / (2.0 * vol)
            else:
                nm = n1 * n2 * sh / vol
            with np.errstate(divide="ignore", invalid="ignore"):
                rdf[f"{s1}-{s2}"] = (rm, np.where(nm > 0, c / nm, 0))
    return rdf

# ---------------------------------------------------------------------------
# Faber-Ziman total RDF weighting (number / X-ray / electron / neutron)
# ---------------------------------------------------------------------------
# A "total" RDF is a weighted sum of the partials g_AB(r):
#
#     G_w(r) = Σ_{A<=B} w_AB · g_AB(r)
#     w_AB   = (2 - δ_AB) · c_A c_B x_A x_B / ( Σ_C c_C x_C )²
#
# where c_A is the number fraction of element A in the structure and x_A is
# the element's scattering coefficient for the chosen radiation.  The
# (2 - δ_AB) factor counts each unordered cross pair twice.  For number
# weighting (x≡1) the weights obey Σ w_AB = 1, so G→1 at large r.
#
# Neutron bound coherent scattering lengths b_coh (fm), natural isotopic
# abundance (Sears, Neutron News 3 (1992) 26; NIST).  Ti/V/Mn/H are genuinely
# NEGATIVE — keep the sign; it makes some neutron weights negative, which is
# correct physics, not a bug.
NEUTRON_B_COH = {
    'H': -3.739, 'Li': -1.90, 'B': 5.30, 'C': 6.646, 'N': 9.36, 'O': 5.803,
    'F': 5.654, 'Na': 3.63, 'Mg': 5.375, 'Al': 3.449, 'Si': 4.1491,
    'P': 5.13, 'S': 2.847, 'Cl': 9.577, 'Ar': 1.909, 'K': 3.67, 'Ca': 4.70,
    'Ti': -3.438, 'V': -0.3824, 'Cr': 3.635, 'Mn': -3.73, 'Fe': 9.45,
    'Co': 2.49, 'Ni': 10.3, 'Cu': 7.718, 'Zn': 5.680, 'Ga': 7.288,
    'Ge': 8.185, 'As': 6.58, 'Se': 7.970, 'Br': 6.795, 'Y': 7.75,
    'Zr': 7.16, 'Nb': 7.054, 'Mo': 6.715, 'Ba': 5.07, 'La': 8.24,
    'Hf': 7.77, 'Ta': 6.91, 'W': 4.86, 'Pt': 9.60, 'Au': 7.63,
    'Pb': 9.405, 'Bi': 8.532,
}

# Forward electron scattering factors f_e(0) (Å), neutral atoms, computed from
# Kirkland's 2010 parameterization (f_e(0) = Σ a_i/b_i + Σ c_i), cross-checked
# against the Peng 5-Gaussian fits (agreement < 1%).  Note f_e(0) is NOT
# monotone in Z (Ti > Ge > Fe; Ar < Si) — it tracks Z·⟨r²⟩ via Mott-Bethe, so
# a power-law-in-Z stand-in would misweight light/heavy element contrast.
ELECTRON_FE0 = {
    'H': 0.5297, 'Li': 3.2898, 'B': 2.7982, 'C': 2.5114, 'N': 2.2188, 'O': 1.9897,
    'F': 1.805, 'Na': 4.7757, 'Mg': 5.2062, 'Al': 5.8919, 'Si': 5.8143, 'P': 5.4976,
    'S': 5.1721, 'Cl': 4.8734, 'Ar': 4.5924, 'K': 8.9466, 'Ca': 9.9096, 'Ti': 8.74,
    'V': 8.2509, 'Cr': 6.9684, 'Mn': 7.4754, 'Fe': 7.1632, 'Co': 6.8529, 'Ni': 6.5572,
    'Cu': 5.5807, 'Zn': 6.0727, 'Ga': 7.1035, 'Ge': 7.3827, 'As': 7.3386, 'Se': 7.2231,
    'Br': 7.0714, 'Y': 12.6753, 'Zr': 12.1895, 'Nb': 10.716, 'Mo': 10.2895, 'Ba': 18.1687,
    'La': 17.7912, 'Hf': 13.0475, 'Ta': 12.7403, 'W': 12.4822, 'Pt': 10.8066, 'Au': 10.5534,
    'Pb': 13.0404, 'Bi': 13.0367,
}

def _fz_weight_coeff(sym, kind):
    """Per-element Faber-Ziman scattering coefficient x_A for `kind`.

    number   : x = 1            (plain number-density total g(r))
    xray     : x = Z            (forward form-factor limit f(Q->0)=Z)
    electron : x = f_e(0) (Å) from ELECTRON_FE0 (Kirkland parameterization)
    neutron  : x = b_coh (fm) from NEUTRON_B_COH (can be negative)

    Returns None when a coefficient is unavailable (e.g. a neutron b or
    electron f_e(0) for an untabulated element) so the caller can mask that
    total channel.
    """
    from ase.data import atomic_numbers
    if kind == 'number':
        return 1.0
    z = atomic_numbers.get(sym)
    if z is None:
        return None
    if kind == 'xray':
        return float(z)
    if kind == 'electron':
        return ELECTRON_FE0.get(sym)
    if kind == 'neutron':
        return NEUTRON_B_COH.get(sym)
    raise ValueError(f"unknown Faber-Ziman weight kind {kind!r}")

def rdf_pair_order(elements):
    """Canonical unordered element-pair keys for a fixed element set.

    Uses the SAME ``sorted(set(...))`` ordering ``comp_rdf`` uses to name its
    keys, so a pair present in a structure's ``comp_rdf`` dict maps onto the
    identical ``"A-B"`` string here.  The result is the fixed channel layout
    for the dense partial-RDF tensor (missing pairs are zero rows).
    """
    sp = sorted(set(elements))
    return [f"{sp[i]}-{sp[j]}" for i in range(len(sp)) for j in range(i, len(sp))]

def rdf_partial_dense(partials, pair_order, nb):
    """Pack a ``comp_rdf`` dict into a dense ``[n_pairs, nb]`` array.

    ``pair_order`` is a fixed global list from ``rdf_pair_order``; pairs
    absent from ``partials`` become zero rows (physically g≡0 for that cell),
    guaranteeing a composition-independent, fixed-shape layout.
    """
    dense = np.zeros((len(pair_order), nb), dtype=np.float32)
    for i, key in enumerate(pair_order):
        pg = partials.get(key)
        if pg is not None:
            dense[i] = pg[1]
    return dense

def faber_ziman(partials, counts, kind='number'):
    """Combine partial g_AB(r) into one Faber-Ziman total RDF.

    partials : dict {"A-B": (rm, g_AB)} as returned by ``comp_rdf``.
    counts   : dict {symbol: n_atoms} for THIS structure (sets the number
               fractions c_A).
    kind     : 'number' | 'xray' | 'electron' | 'neutron', or a dict
               {symbol: x_A} of explicit coefficients.

    Returns ``(rm, G)`` or ``None`` when a required coefficient is missing
    (e.g. a neutron b_coh for an untabulated element) so the caller can leave
    that channel non-conditioned.
    """
    if not partials:
        return None
    elems = sorted(counts)
    ntot = float(sum(counts[e] for e in elems))
    if ntot <= 0:
        return None
    if isinstance(kind, dict):
        xco = {e: kind.get(e) for e in elems}
    else:
        xco = {e: _fz_weight_coeff(e, kind) for e in elems}
    if any(xco.get(e) is None for e in elems):
        return None
    c = {e: counts[e] / ntot for e in elems}
    denom = sum(c[e] * xco[e] for e in elems) ** 2
    if denom == 0:
        return None
    rm = None; G = None
    for key, (r, g) in partials.items():
        a, b = key.split('-')
        if a not in c or b not in c:
            continue
        delta = 1.0 if a == b else 2.0
        w = delta * c[a] * c[b] * xco[a] * xco[b] / denom
        if G is None:
            rm = r; G = np.zeros_like(g, dtype=float)
        G = G + w * g
    if G is None:
        return None
    return rm, G

###############################################################################
# PLOTS
###############################################################################
def plt_lim(atoms, rot="45x,45y,0z", mg=.08):
    n = len(atoms); probe = atoms
    if n > _LT: rng = np.random.default_rng(42); probe = atoms[np.sort(rng.choice(n,min(n,500),replace=False))]
    fig,ax = plt.subplots(); plot_atoms(probe,ax,rotation=rot)
    x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim(); plt.close(fig)
    if n > _LT:
        dg = np.linalg.norm(atoms.cell.lengths()); sx,sy = max(x1-x0,1),max(y1-y0,1)
        sc = max(dg/sx,dg/sy,1); mx,my = .5*(x0+x1),.5*(y0+y1)
        return (mx-.5*sx*sc*(1+mg),mx+.5*sx*sc*(1+mg),my-.5*sy*sc*(1+mg),my+.5*sy*sc*(1+mg))
    xp,yp = (x1-x0)*mg,(y1-y0)*mg; return (x0-xp,x1+xp,y0-yp,y1+yp)

def rnd_atoms(atoms, ax, lim=None, title=None, rot="45x,45y,0z", mx=5000):
    n = len(atoms)
    if n > mx:
        rng = np.random.default_rng(42); plot_atoms(atoms[np.sort(rng.choice(n,mx,replace=False))],ax,rotation=rot)
        if title: title = f"{title} ({mx}/{n})"
    else: plot_atoms(atoms,ax,rotation=rot)
    if lim: ax.set_xlim(lim[0],lim[1]); ax.set_ylim(lim[2],lim[3])
    ax.set_aspect("equal",adjustable="box"); ax.axis("off")
    if title: ax.set_title(title,fontsize=9)
    # Legend handles MUST come from the same color source ASE.plot_atoms
    # uses to draw the atoms (jmol_colors), via _atom_color. Otherwise
    # elements like Ta, Zr, Si — which weren't in the old hand-rolled
    # CPK — get a pink "default" patch in the legend while their atoms
    # are drawn in their real jmol colors.
    ax.legend(handles=[mpatches.Patch(facecolor=_atom_color(s),edgecolor="#333",lw=.5,label=s)
              for s in sorted(set(atoms.get_chemical_symbols()))],loc="upper right",fontsize=8,framealpha=.7)

def plt_cn(cd, cuts):
    ml = [e for e,cn in cd.items() if len(cn)]
    if not ml: return None
    fig,axes = plt.subplots(1,len(ml),figsize=(4.5*len(ml),4),squeeze=False)
    for ax,el in zip(axes[0],ml):
        cn = cd[el]; bins = np.arange(cn.min(),cn.max()+2)-.5
        ch,_,pa = ax.hist(cn,bins=bins,color="#4C9BE8",edgecolor="white",lw=.5)
        for c,p in zip(ch,pa):
            if c: ax.text(p.get_x()+p.get_width()/2,p.get_height()+len(cn)*.01,f"{100*c/len(cn):.1f}%",ha="center",va="bottom",fontsize=8)
        ax.set_title(f"{el}-O (cut {cuts.get(el,'?')}Å)\nmean={cn.mean():.2f} std={cn.std():.2f}")
        ax.set_xlabel("CN(O)"); ax.set_ylabel("Count")
    fig.tight_layout(); return fig

def plt_rdf(rd):
    if not rd: return None
    pa = list(rd.items()); nc = min(3,len(pa)); nr = int(np.ceil(len(pa)/nc))
    fig,axes = plt.subplots(nr,nc,figsize=(5*nc,3.5*nr),squeeze=False)
    for i,(p,(r,g)) in enumerate(pa):
        ax = axes.flatten()[i]; ax.plot(r,g,color=plt.cm.tab10.colors[i%10],lw=1.3)
        ax.axhline(1,color="gray",ls="--",lw=.8,alpha=.6); ax.set_xlabel("r(Å)"); ax.set_ylabel("g(r)")
        ax.set_title(p); ax.set_xlim(0,r[-1]); ax.set_ylim(bottom=0)
    for ax in axes.flatten()[len(pa):]: ax.set_visible(False)
    fig.suptitle("Partial RDFs",fontsize=12,y=1.01); fig.tight_layout(); return fig

###############################################################################
# HELPERS
###############################################################################
def safe_rmax(atoms):
    """Maximum safe RDF radius: half the inscribed-sphere diameter of the cell.
    For orthogonal cells this is min(a,b,c)/2.
    For general cells it accounts for off-diagonal terms."""
    cell = np.array(atoms.cell)
    # Inscribed sphere radius = min distance from origin to each face
    # Face normals are cross products of cell vector pairs, distance = V / |face_area|
    a, b, c = cell[0], cell[1], cell[2]
    V = abs(np.dot(a, np.cross(b, c)))
    if V < 1e-12:
        return float(min(atoms.cell.lengths())) / 2
    h_a = V / np.linalg.norm(np.cross(b, c))
    h_b = V / np.linalg.norm(np.cross(a, c))
    h_c = V / np.linalg.norm(np.cross(a, b))
    return min(h_a, h_b, h_c) / 2.0
