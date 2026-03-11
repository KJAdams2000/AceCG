# AceCG/trainers/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List, Union
from pathlib import Path
import os

import numpy as np
import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils.neighbor import Pair2DistanceByFrame, combine_Pair2DistanceByFrame
from ..utils.trjio import split_lammpstrj


def optimizer_accepts_hessian(optimizer) -> bool:
    """
    Check whether the optimizer's `step` method accepts a 'hessian' argument.

    Parameters
    ----------
    optimizer : object
        An optimizer instance with a `.step()` method.

    Returns
    -------
    bool
        True if 'hessian' is a parameter of the .step() method, else False.
    """
    return hasattr(optimizer, 'step') and 'hessian' in optimizer.step.__code__.co_varnames


def _worker_pair2dist(
    topology: Optional[str],
    trj_path: str,
    cutoff: float,
    pair2potential: Dict[Tuple[str, str], Any],
    sel: str,
    nstlist: int,
    exclude: bool,
) -> Dict[int, Dict[Tuple[str, str], np.ndarray]]:
    """
    Worker: load a per-chunk Universe and compute Pair2DistanceByFrame for all frames in that chunk.
    The returned dict uses *local* frame indices (0..n_frames-1) within the chunk.
    """
    if topology is None:
        u = mda.Universe(trj_path, format="LAMMPSDUMP")
    else:
        u = mda.Universe(topology, trj_path, format="LAMMPSDUMP")

    return Pair2DistanceByFrame(
        u,
        start=0,
        end=len(u.trajectory),
        cutoff=cutoff,
        pair2potential=pair2potential,
        sel=sel,
        nstlist=nstlist,
        exclude=exclude,
    )


def prepare_Trainer_data(
    u,
    pair2potential: Dict[Tuple[str, str], Any],
    start: int,
    end: int,
    cutoff: float,
    sel: str = "all",
    nstlist: int = 10,
    exclude: bool = True,
    weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract pairwise distances from an MDAnalysis Universe into Trainer format.

    This is the serial version. It directly analyzes the trajectory contained
    in the provided ``MDAnalysis.Universe``.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Input simulation system.
    pair2potential : dict
        Mapping of (type1, type2) to potential objects.
    start : int
        Starting frame index (inclusive).
    end : int
        Ending frame index (exclusive).
    cutoff : float
        Neighbor cutoff.
    sel : str
        Atom selection.
    nstlist : int
        Neighbor list update interval.
    exclude : bool
        Whether to exclude bonded neighbors.
    weight : np.ndarray, optional
        Optional per-frame weight array.

    Returns
    -------
    dict
        {
            "dist": Pair2DistanceByFrame output,
            "weight": weight
        }
    """

    dist = Pair2DistanceByFrame(
        u,
        start=start,
        end=end,
        cutoff=cutoff,
        pair2potential=pair2potential,
        sel=sel,
        nstlist=nstlist,
        exclude=exclude,
    )

    return {"dist": dist, "weight": weight}


def prepare_Trainer_data_parallel(
    *,
    topology: Optional[Union[str, Path]] = None,
    trajectory: Union[str, Path],
    pair2potential: Dict[Tuple[str, str], Any],
    start: int,
    end: int,
    cutoff: float,
    sel: str = "all",
    nstlist: int = 10,
    exclude: bool = True,
    weight: Optional[np.ndarray] = None,
    n_parts: int = 8,
    n_workers: Optional[int] = None,
    chunk_dir: Union[str, Path] = "traj_chunks",
    chunk_prefix: str = "chunk",
    keep_chunks: bool = True,
) -> Dict[str, Any]:
    """
    Parallel version of prepare_Trainer_data.

    This function splits a trajectory file into smaller chunk trajectories,
    computes Pair2DistanceByFrame for each chunk in parallel, and then
    combines the results into a single frame-indexed dictionary.

    Unlike the serial version, this function does NOT accept an MDAnalysis
    Universe. Workers reopen trajectory chunks independently.

    Parameters
    ----------
    topology : str or Path, optional
        Topology file used to construct MDAnalysis Universe objects.
    trajectory : str or Path
        Input trajectory path.
    pair2potential : dict
        Mapping of (type1, type2) to potential objects.
    start : int
        Starting frame index.
    end : int
        Ending frame index.
    cutoff : float
        Neighbor cutoff.
    sel : str
        Atom selection.
    nstlist : int
        Neighbor list update interval.
    exclude : bool
        Whether to exclude bonded neighbors.
    weight : np.ndarray, optional
        Frame weights.
    n_parts : int
        Number of trajectory chunks.
    n_workers : int, optional
        Number of worker processes.
    chunk_dir : str or Path
        Directory for temporary chunk trajectories.
    chunk_prefix : str
        Prefix for chunk filenames.
    keep_chunks : bool
        Whether to keep chunk files.

    Returns
    -------
    dict
        {
            "dist": combined Pair2DistanceByFrame,
            "weight": weight[start:end] or None
        }
    """

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    trajectory = Path(trajectory)
    if not trajectory.exists():
        raise FileNotFoundError(trajectory)

    if start < 0:
        raise ValueError("start must be >= 0")

    if end <= start:
        raise ValueError("end must be > start")

    topo_str = str(topology) if topology is not None else None

    if weight is not None:
        weight = np.asarray(weight)
        if len(weight) < end:
            raise ValueError("weight shorter than trajectory slice")
        weight_out = weight[start:end]
    else:
        weight_out = None

    chunk_dir = Path(chunk_dir)

    chunk_paths = split_lammpstrj(
        trj_path=trajectory,
        n_parts=n_parts,
        out_dir=chunk_dir,
        prefix=chunk_prefix,
        digits=3,
        dry_run=False,
    )

    chunk_meta = []
    gcur = 0

    for p in chunk_paths:

        if topo_str is None:
            uu = mda.Universe(str(p), format="LAMMPSDUMP")
        else:
            uu = mda.Universe(topo_str, str(p), format="LAMMPSDUMP")

        nfr = len(uu.trajectory)

        chunk_meta.append((gcur, gcur + nfr, p))

        gcur += nfr

    if start >= gcur:
        raise ValueError("start beyond trajectory length")

    if end > gcur:
        raise ValueError("end beyond trajectory length")

    use_chunks = []

    for g0, g1, p in chunk_meta:

        if g1 <= start or g0 >= end:
            continue

        use_chunks.append((g0, g1, p))

    results_by_chunk_start = {}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:

        futs = {}

        for g0, g1, p in use_chunks:

            fut = ex.submit(
                _worker_pair2dist,
                topo_str,
                str(p),
                cutoff,
                pair2potential,
                sel,
                nstlist,
                exclude,
            )

            futs[fut] = (g0, g1, p)

        for fut in as_completed(futs):

            g0, g1, p = futs[fut]

            d_local = fut.result()

            results_by_chunk_start[g0] = d_local

    dicts_to_combine = []

    for g0, g1, p in sorted(use_chunks, key=lambda x: x[0]):

        d_local = results_by_chunk_start[g0]

        lo = max(start, g0) - g0
        hi = min(end, g1) - g0

        d_slice = {i: d_local[i] for i in range(lo, hi)}

        dicts_to_combine.append(d_slice)

    dist = combine_Pair2DistanceByFrame(
        dicts_to_combine,
        start_frame=start,
    )

    if not keep_chunks:

        for _, _, p in chunk_meta:

            try:
                p.unlink()
            except Exception:
                pass

        try:
            chunk_dir.rmdir()
        except Exception:
            pass

    return {"dist": dist, "weight": weight_out}