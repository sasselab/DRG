#!/usr/bin/env python
"""Tn5 insertion-site counts from paired-end ATAC-seq BAM files.

Reads a coordinate-sorted, indexed BAM and counts Tn5 insertion sites (or
fragment coverage) at single base-pair resolution, either per region of a bed
file (npz) or over the whole genome (bigWig).

====================================================================
General usage for ATAC-seq
====================================================================
    # 1. Genome-wide Tn5 insertion track (no --bed), unstranded bigWig
    python process_atac_bp.py sample.bam --outdir bigwigs/ -p 8

    # 2. Same, in counts per million, plus/minus strand tracks, main chromosomes
    python process_atac_bp.py sample.bam --outdir bigwigs/ -p 8 --cpm --stranded \
        --chroms chr1 chr2 ... chrX

    # 3. Per-region profiles for a bed file (npz), 1000 bp windows around peak centers
    python process_atac_bp.py sample.bam --bed peaks.txt --length 1000 --outdir BPprofiles/ -p 8

    # 4. Same with standard BED (contig start end [name]), blacklist, 10 bp bins
    python process_atac_bp.py sample.bam --bed peaks.bed --standard-bed --length 1000 \
        --blacklist blacklist.bed --meancounts 10 --outdir BPprofiles/

Input BAM: ideally filtered as in the ENCODE ATAC pipeline (MAPQ >= 30,
duplicates removed/marked, mitochondrial reads removed), i.e. what
``samtools view -b -F 780 -q 30`` gives. Duplicate-flagged, secondary,
supplementary and QC-fail alignments are skipped here; use ``--min-mapq`` for a
MAPQ filter. Tn5 binds as a dimer and cuts with a 9 bp stagger, so ends of the
fragment are shifted by +4 (left/plus strand) and -4 (right/minus strand) bp
to centre the insertion on the 9 bp site (``--shift 4,-4``, the default).

====================================================================
Relation to ChromBPNet / gReLU
====================================================================
ChromBPNet (kundajelab/chrombpnet, ``helpers/preprocessing/reads_to_bigwig.py``)
builds the training bigWig of 5' ends from a BAM with this shell pipeline:

    bam -> tagAlign stream (bedtools bamtobed / auto_shift_detect)
        | awk '{if ($6=="+") print $1,$2+PS,$3,...; else print $1,$2,$3+MS,...}'
        | sort -k1,1
        | bedtools genomecov -bg -5 -i stdin -g chrom.sizes
        | LC_COLLATE=C sort -k1,1 -k2,2n   > unstranded.bedGraph
    bedGraphToBigWig unstranded.bedGraph chrom.sizes prefix_unstranded.bw

with PS/MS = +4/-4 for ATAC (estimated from reference motifs if not given).
gReLU (``grelu.data.preprocess.make_insertion_bigwig``) runs the same chain
starting from a fragment file: for every fragment it emits the start shifted by
``plus_shift`` and the end shifted by ``minus_shift``, then genomecov -5 and
bedGraphToBigWig. This script reproduces both without external tools:

    ChromBPNet:  --readmode read     --countmode 5 --shift 4,-4   (5' read ends)
    gReLU:       --readmode fragment --countmode 53 --shift 4,-4  (fragment ends)

Checked on ENCODE ENCFF613OUT (GRCh38): identical to the bedtools pipelines at
every base on chr1, chr10, chr21 and chrX. It is not a drop-in replacement for
the automatic shift estimation of ChromBPNet (give ``--shift`` explicitly) and
it writes the bigWig directly (no bedGraph/sort step), in parallel by tiles.

====================================================================
Parameters
====================================================================
Input / output
    bam                 sorted, indexed BAM (positional).
    --bed FILE          regions to count. If omitted, a genome-wide bigWig is
                        written instead of an npz.
    --standard-bed      --bed/--blacklist are standard BED (contig start end
                        [name]) instead of the DRG format (name contig start end).
    --oneindex          bed coordinates are 1-based (1 is subtracted).
    --outdir DIR        output directory (default: .).
    --outname NAME      output file prefix (default: built from file names and
                        options). npz mode appends ``.npz`` (``sl<LEN>`` with
                        --length, ``avg<N>`` with --meancounts); bigWig mode
                        writes ``NAME.bw`` or ``NAME_plus.bw``/``NAME_minus.bw``.

What is counted
    --readmode {fragment,read}
                        fragment (default): properly paired reads define the
                        fragment (start of left mate to end of right mate).
                        read: every alignment is used on its own (single-end data).
    --countmode {53,5,3,entire,entirefraction}
                        53 (default): both fragment ends = Tn5 insertion sites.
                        5 / 3: only the 5' / 3' end (fragment mode: left / right
                        end; read mode: by strand of the read). entire: +1 on
                        every base of the fragment. entirefraction: +1/length
                        on every base (one weight per fragment).
    --shift LEFT,RIGHT  shift of left and right end in bp (default 4,-4).
    --stranded          count plus/minus separately (53/5/3 only). Fragment mode:
                        left end = plus, right end = minus; read mode: strand of
                        the read. bigWig: ``_plus``/``_minus`` files; npz: 2 channels.
    --maxlen N          skip fragments/reads longer than N bp.
    --min-mapq N        skip alignments with MAPQ < N (default 0).
    --keep-duplicates   also count duplicate-flagged alignments (skipped by default).
    --blacklist FILE    skip fragments with a shifted end inside these regions.
    --fetch-pad N       entire/entirefraction without --maxlen: fragments whose
                        left mate lies more than N bp left of a window are missed
                        (default 2000). Insertion modes need no padding.

Region mode (--bed)
    --length LEN        every region is trimmed/extended symmetrically around its
                        center to LEN bp. Default: length of the longest region.
                        A warning is shown if regions differ in size; windows over
                        a chromosome end are shifted inward, so there is never
                        zero padding and every position is a real count.
    --meancounts N      mean-pool in bins of N bp (N must divide the window
                        length); 0 averages the whole window to one value.
    --batch-size N      regions per task (default 500).

Genome mode (no --bed)
    --chroms C [C ...]  only these chromosomes.
    --cpm               scale to counts per million mapped reads (of the
                        selected chromosomes; fragment mode counts two reads
                        per fragment).
    --tile-size N       tile size in bp for parallel counting (default 5 Mb).

General
    -p, --processes N   worker processes (default: all CPUs).
    -v, --verbose       debug logging.

====================================================================
Outputs
====================================================================
Genome mode: bigWig(s) at 1 bp resolution, equal neighbouring values merged.

Region mode: npz with ``names`` (regions kept) and ``counts`` of shape
(n_regions, channels, n_positions). ``channels`` is 1, or 2 (plus, minus) with
--stranded, so npz files can be concatenated along that axis. n_positions is the
window length (1 bp resolution) or window_length / N with --meancounts N (mean,
not sum, per bin). Regions on contigs missing from the BAM are dropped with a
warning.

Details
    * Each end of a fragment is counted from its own mate, so a fragment is
      counted once even if mates start at the same position or the fragment is
      much longer than the window.
    * Single-end BAMs are detected (no proper pairs) and the script stops with
      a hint to use --readmode read.
    * Window counting is vectorised with numpy; whole-fragment modes use a
      difference array, and nearby bed regions share one BAM fetch.
"""
import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pysam

logger = logging.getLogger("process_atac_bp")

COUNT_MODES = ("53", "5", "3", "entire", "entirefraction")
SPARSE_MODES = ("53", "5", "3")
MAX_CLUSTER = 100_000  # max span (bp) of bed regions sharing one BAM fetch


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def read_bed(bedfile, offset=0, named=True):
    """Read a region file.

    Args:
        bedfile: path to the file (whitespace separated, '#' lines ignored).
        offset: value subtracted from start and end (use 1 for 1-based files).
        named: if True, columns are ``name contig start end`` (DRG format);
            otherwise standard BED ``contig start end [name]``.

    Returns:
        dict with arrays ``names``, ``contigs`` (str) and ``positions``
        (n, 2) int64 array.
    """
    names, contigs, positions = [], [], []
    with open(bedfile) as fh:
        for i, line in enumerate(fh):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            if named:
                name, contig, start, end = f[0], f[1], f[2], f[3]
            else:
                contig, start, end = f[0], f[1], f[2]
                name = f[3] if len(f) > 3 else f"{contig}:{start}-{end}"
            names.append(name)
            contigs.append(contig)
            positions.append((int(start) - offset, int(end) - offset))
    return {
        "names": np.array(names),
        "contigs": np.array(contigs),
        "positions": np.array(positions, dtype=np.int64).reshape(-1, 2),
    }


def merge_intervals(contigs, positions):
    """Merge overlapping intervals per contig.

    Returns:
        dict contig -> (starts, ends) sorted, non-overlapping int64 arrays.
    """
    merged = {}
    for chrom in np.unique(contigs):
        iv = positions[contigs == chrom]
        iv = iv[np.argsort(iv[:, 0])]
        starts, ends = [iv[0, 0]], [iv[0, 1]]
        for s, e in iv[1:]:
            if s <= ends[-1]:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        merged[str(chrom)] = (np.array(starts), np.array(ends))
    return merged


def in_intervals(pos, intervals):
    """Boolean mask: which positions in ``pos`` lie in merged [start, end) intervals."""
    if intervals is None:
        return np.zeros(len(pos), dtype=bool)
    starts, ends = intervals
    idx = np.searchsorted(starts, pos, side="right") - 1
    return (idx >= 0) & (pos < ends[np.clip(idx, 0, None)])


def get_total_mapped(bamfile, chroms=None):
    """Number of mapped reads from the BAM index (idxstats), before any filtering.

    Args:
        chroms: optional collection of contigs to restrict the count to.
    """
    total = 0
    for line in pysam.idxstats(bamfile).splitlines():
        f = line.split("\t")
        if len(f) >= 3 and (not chroms or f[0] in chroms):
            total += int(f[2])
    return total


def check_paired(bamfile, n=5000):
    """Return True if any of the first ``n`` mapped reads is a proper pair."""
    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if read.is_proper_pair:
                return True
            if i >= n:
                break
    return False


# --------------------------------------------------------------------------- #
# Core counting
# --------------------------------------------------------------------------- #
def fetch_rows(bam, chrom, start, stop, opts):
    """Fetch alignments in [start, stop) as an int64 array of rows (st, en, role, rev).

    ``st``/``en`` are the (unshifted, ``en`` exclusive) coordinates of the
    fragment (fragment mode) or of the read (read mode).
    ``role`` is 0 for the left mate, 1 for the right mate of a fragment (the
    left mate is the one with the smaller start, read 1 on ties) and 2 for a
    single read. ``rev`` is the strand of the alignment (1 = minus).

    Every mate of a fragment is returned (each one carries the fragment
    coordinates from its template length), so fragments much longer than the
    window are still found through whichever mate lies in the window.
    """
    rows = []
    minq, maxlen = opts["min_mapq"], opts["maxlen"]
    fragment = opts["readmode"] == "fragment"
    for read in bam.fetch(chrom, start, stop):
        if (read.is_unmapped or read.is_secondary or read.is_supplementary
                or read.is_qcfail or read.mapping_quality < minq
                or (read.is_duplicate and not opts["keep_duplicates"])):
            continue
        if fragment:
            tlen = abs(read.template_length)
            if not read.is_proper_pair or tlen == 0:
                continue
            mate = read.next_reference_start
            left = read.reference_start < mate or (read.reference_start == mate and read.is_read1)
            st = read.reference_start if left else mate
            en = st + tlen
            role, rev = (0, 0) if left else (1, 1)
        else:
            st, en = read.reference_start, read.reference_end
            role, rev = 2, int(read.is_reverse)
        if maxlen is not None and en - st > maxlen:
            continue
        rows.append((st, en, role, rev))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def accumulate(rows, win_start, win_end, opts, blacklist=None):
    """Turn fetched rows into counts for the window [win_start, win_end).

    Args:
        rows: array from :func:`fetch_rows`.
        opts: dict with readmode, countmode, shift, stranded.
        blacklist: merged intervals (starts, ends) of the contig or None.
            Fragments with an insertion site in the blacklist are skipped.

    Returns:
        Array of shape (n_strands, window_length); n_strands is 2 (plus, minus)
        if ``opts['stranded']`` else 1. A (fragment-mode) insertion is on the
        plus strand if it is the left end and on the minus strand if it is the
        right end; in read mode it has the strand of the read.
    """
    cmode, readmode = opts["countmode"], opts["readmode"]
    sh0, sh1 = opts["shift"]
    length = win_end - win_start
    nstr = 2 if opts["stranded"] else 1
    out = np.zeros((nstr, length), dtype=np.float32 if cmode == "entirefraction" else np.int32)
    if len(rows) == 0:
        return out

    st = rows[:, 0] + sh0
    en = rows[:, 1] + sh1
    role, rev = rows[:, 2], rows[:, 3].astype(bool)
    fraglen = rows[:, 1] - rows[:, 0]

    # only rows that can touch the window (insertion or covered bases)
    near = (en > win_start - 1) & (st < win_end + 1)
    if blacklist is not None:
        near &= ~(in_intervals(st, blacklist) | in_intervals(en - 1, blacklist))
    st, en, role, rev, fraglen = st[near], en[near], role[near], rev[near], fraglen[near]

    if cmode in SPARSE_MODES:
        use_left, use_right = role != 1, role != 0
        fragment = readmode == "fragment"
        if cmode == "5":
            use_left, use_right = (use_left, use_right & False) if fragment else (use_left & ~rev, use_right & rev)
        elif cmode == "3":
            use_left, use_right = (use_left & False, use_right) if fragment else (use_left & rev, use_right & ~rev)
        pos = np.concatenate([st[use_left], en[use_right] - 1]) - win_start
        strand = np.concatenate([rev[use_left], rev[use_right]]) if nstr == 2 else np.zeros(len(pos), bool)
        ok = (pos >= 0) & (pos < length)
        for s in range(nstr):
            np.add.at(out[s], pos[ok & (strand == bool(s))], 1)
    else:
        # one count per fragment: use the left mate only
        keep = role != 1
        st, en, fraglen = st[keep], en[keep], fraglen[keep]
        weight = np.ones(len(st)) if cmode == "entire" else 1.0 / fraglen
        s = np.clip(st - win_start, 0, length)
        e = np.clip(en - win_start, 0, length)
        valid = e > s
        # difference array makes coverage O(n) instead of O(n * fragment length)
        diff = np.zeros(length + 1, dtype=np.float64)
        np.add.at(diff, s[valid], weight[valid])
        np.add.at(diff, e[valid], -weight[valid])
        cov = np.cumsum(diff)[:length]
        out[0] = np.rint(cov) if cmode == "entire" else cov
    return out


def count_window(bam, chrom, win_start, win_end, chrom_len, opts, blacklist=None):
    """Count insertions/coverage in [win_start, win_end) of ``chrom``.

    Returns an array of shape (n_strands, window_length), see :func:`accumulate`.
    """
    lo, hi = max(0, win_start - opts["pad"]), min(chrom_len, win_end + opts["pad"])
    rows = fetch_rows(bam, chrom, lo, hi, opts) if lo < hi else np.zeros((0, 4), np.int64)
    return accumulate(rows, win_start, win_end, opts, blacklist)


# --------------------------------------------------------------------------- #
# Multiprocessing workers (each opens its own BAM handle)
# --------------------------------------------------------------------------- #
_STATE = {}


def _init_worker(bampath, opts, blacklist):
    _STATE["bam"] = pysam.AlignmentFile(bampath, "rb")
    _STATE["opts"] = opts
    _STATE["blacklist"] = blacklist


def _bed_task(task):
    """Count a batch of bed regions on one chromosome. Returns (indices, counts).

    Nearby regions share a single BAM fetch (clusters of at most
    ``MAX_CLUSTER`` bp) so overlapping or adjacent regions don't re-read reads.
    """
    chrom, indices, positions, seqlen = task
    bam, opts = _STATE["bam"], _STATE["opts"]
    dtype = np.float32 if opts["countmode"] == "entirefraction" else np.int32
    nstr = 2 if opts["stranded"] else 1
    counts = np.zeros((len(indices), nstr, seqlen), dtype=dtype)
    clen = bam.get_reference_length(chrom)
    bl = (_STATE["blacklist"] or {}).get(chrom)
    pad = opts["pad"]

    order = np.argsort(positions[:, 0], kind="stable")
    k = 0
    while k < len(order):
        first = order[k]
        members = [first]
        c_start, c_end = positions[first]
        k += 1
        while k < len(order) and positions[order[k], 1] - c_start <= MAX_CLUSTER \
                and positions[order[k], 0] <= c_end + pad:
            members.append(order[k])
            c_end = max(c_end, positions[order[k], 1])
            k += 1
        rows = fetch_rows(bam, chrom, max(0, c_start - pad), min(clen, c_end + pad), opts)
        for m in members:
            counts[m] = accumulate(rows, int(positions[m, 0]), int(positions[m, 1]), opts, bl)
    return indices, counts


def run_length_encode(vals, offset):
    """Collapse a 1D array into runs of equal non-zero values.

    Returns (starts, ends, values) with ``ends`` exclusive and coordinates
    shifted by ``offset``; zero stretches are omitted.
    """
    nz = np.flatnonzero(vals)
    if len(nz) == 0:
        return nz, nz, vals[:0]
    v = vals[nz]
    breaks = np.flatnonzero((np.diff(nz) != 1) | (np.diff(v) != 0)) + 1
    first = np.concatenate([[0], breaks])
    last = np.concatenate([breaks - 1, [len(nz) - 1]])
    return nz[first] + offset, nz[last] + 1 + offset, v[first]


def _genome_task(task):
    """Count one genome tile. Returns (chrom, [(starts, ends, values) per strand])."""
    chrom, start, end = task
    bam, opts = _STATE["bam"], _STATE["opts"]
    bl = (_STATE["blacklist"] or {}).get(chrom)
    vals = count_window(bam, chrom, start, end, bam.get_reference_length(chrom), opts, bl)
    return chrom, [run_length_encode(v, start) for v in vals]


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def pool_bins(counts, window):
    """Mean-pool the last axis of an array in non-overlapping bins of ``window``.

    The length of the last axis must be divisible by ``window`` (no padding).
    """
    if counts.shape[-1] % window:
        raise ValueError(f"window length {counts.shape[-1]} is not divisible by bin size {window}")
    return counts.reshape(*counts.shape[:-1], -1, window).mean(axis=-1)


def resize_regions(positions, contigs, chrom_lengths, length=None):
    """Give all regions the same length by trimming/extending around their centers.

    Windows that would cross a chromosome boundary are shifted inward, so every
    position of every window lies on the chromosome and is actually counted.

    Args:
        positions: (n, 2) array of start, end.
        contigs: array of contig names, one per region.
        chrom_lengths: dict contig -> length.
        length: target window length; if None the longest region defines it.

    Returns:
        (new_positions, length)
    """
    sizes = positions[:, 1] - positions[:, 0]
    if length is None:
        length = int(sizes.max())
        n_diff = int((sizes != length).sum())
        if n_diff:
            logger.warning(
                "%d of %d regions are shorter than the longest region (%d bp, "
                "min %d bp); they are extended symmetrically to %d bp.",
                n_diff, len(sizes), length, sizes.min(), length)
    else:
        n_diff = int((sizes != length).sum())
        if n_diff:
            logger.warning("%d of %d regions differ from --length %d and are "
                           "trimmed/extended from their center.", n_diff, len(sizes), length)
    clens = np.array([chrom_lengths[c] for c in contigs])
    if (clens < length).any():
        raise ValueError(f"Window length {length} is longer than a chromosome with regions")
    centers = (positions[:, 0] + positions[:, 1]) // 2
    starts = np.clip(centers - length // 2, 0, clens - length)
    n_shift = int((starts != centers - length // 2).sum())
    if n_shift:
        logger.warning("%d windows reach over a chromosome end and were shifted inward "
                       "so that they stay within the chromosome.", n_shift)
    return np.stack([starts, starts + length], axis=1), length


def run_bed(args, opts, blacklist, outname):
    """Region mode: count per bed region and save npz."""
    offset = 1 if args.oneindex else 0
    bed = read_bed(args.bed, offset=offset, named=not args.standard_bed)
    with pysam.AlignmentFile(args.bam, "rb") as bam:
        chrom_lengths = dict(zip(bam.references, bam.lengths))
    present = np.array([c in chrom_lengths for c in bed["contigs"]])
    if not present.all():
        missing = sorted(set(bed["contigs"][~present]))
        logger.warning("Dropping %d regions on contigs not in the BAM: %s",
                       int((~present).sum()), ", ".join(missing[:10]))
        bed = {k: v[present] for k, v in bed.items()}
    if len(bed["names"]) == 0:
        sys.exit("No bed regions left to count")

    pos, seqlen = resize_regions(bed["positions"], bed["contigs"], chrom_lengths, args.length)
    if args.length:
        outname += f"sl{args.length}"
    if args.meancounts and seqlen % args.meancounts:
        sys.exit(f"--meancounts {args.meancounts} must divide the window length {seqlen}")
    logger.info("Bed regions: %d, window length: %d", len(pos), seqlen)

    tasks = []
    for chrom in np.unique(bed["contigs"]):
        idx = np.flatnonzero(bed["contigs"] == chrom)
        for b in range(0, len(idx), args.batch_size):
            sub = idx[b : b + args.batch_size]
            tasks.append((str(chrom), sub, pos[sub], seqlen))

    nstr = 2 if opts["stranded"] else 1
    dtype = np.float32 if opts["countmode"] == "entirefraction" else np.int32
    counts = np.zeros((len(pos), nstr, seqlen), dtype=dtype)
    with Pool(args.processes, _init_worker, (args.bam, opts, blacklist)) as pool:
        for n, (idx, c) in enumerate(pool.imap_unordered(_bed_task, tasks), 1):
            counts[idx] = c
            if n % 20 == 0 or n == len(tasks):
                logger.info("Finished %d/%d batches", n, len(tasks))

    total = counts.sum()
    logger.info("Total counts %.1f (%.2f per region)", total, total / max(len(pos), 1))
    if args.meancounts is not None:
        counts = pool_bins(counts, args.meancounts) if args.meancounts > 0 else counts.mean(-1, keepdims=True)
        outname += f"avg{args.meancounts}"
    np.savez_compressed(outname + ".npz", names=bed["names"], counts=counts)
    logger.info("Saved %s.npz", outname)


def run_genome(args, opts, blacklist, outname):
    """Genome mode: count in tiles in parallel and stream into bigWig file(s)."""
    try:
        import pyBigWig
    except ImportError:
        sys.exit("pyBigWig is required for bigWig output (pip install pyBigWig)")

    with pysam.AlignmentFile(args.bam, "rb") as bam:
        chroms = [(c, l) for c, l in zip(bam.references, bam.lengths) if not args.chroms or c in args.chroms]
    if not chroms:
        sys.exit("No chromosomes selected/found in BAM header")
    scale = 1.0
    if args.cpm:
        total = get_total_mapped(args.bam, {c for c, _ in chroms})
        if args.readmode == "fragment":
            total //= 2  # two mapped reads per fragment
        scale = 1e6 / max(total, 1)
        logger.info("CPM scaling factor %.4g", scale)

    names = [outname + s for s in (("_plus.bw", "_minus.bw") if args.stranded else (".bw",))]
    header = [(c, int(l)) for c, l in chroms]
    bws = [pyBigWig.open(n, "w") for n in names]
    for bw in bws:
        bw.addHeader(header, maxZooms=10)

    tasks = [(c, s, min(s + args.tile_size, l)) for c, l in chroms for s in range(0, l, args.tile_size)]
    # imap preserves order, which bigWig requires (sorted by chromosome, then position)
    with Pool(args.processes, _init_worker, (args.bam, opts, blacklist)) as pool:
        for n, (chrom, per_strand) in enumerate(pool.imap(_genome_task, tasks, chunksize=1), 1):
            for bw, (starts, ends, values) in zip(bws, per_strand):
                if len(starts):
                    bw.addEntries([chrom] * len(starts), starts.astype(np.int64),
                                  ends=ends.astype(np.int64), values=(values * scale).astype(np.float32))
            if n % 50 == 0 or n == len(tasks):
                logger.info("Finished %d/%d tiles", n, len(tasks))
    for bw in bws:
        bw.close()
    logger.info("Saved %s", ", ".join(names))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("bam", help="sorted, indexed paired-end BAM file")
    p.add_argument("--bed", help="region file; if omitted, genome-wide bigWig is written")
    p.add_argument("--standard-bed", action="store_true",
                   help="--bed is standard BED (contig start end [name]) instead of 'name contig start end'")
    p.add_argument("--oneindex", action="store_true", help="bed coordinates are 1-based (subtract 1)")
    p.add_argument("--length", type=int, metavar="LEN",
                   help="(bed mode) fixed window length; every region is trimmed/extended to LEN bp "
                        "around its center (default: length of the longest region)")
    p.add_argument("--outdir", default=".", help="output directory (default: .)")
    p.add_argument("--outname", help="output file prefix (default: derived from input names/options)")
    p.add_argument("--readmode", choices=("fragment", "read"), default="fragment")
    p.add_argument("--countmode", choices=COUNT_MODES, default="53")
    p.add_argument("--maxlen", type=int, help="ignore fragments/reads longer than this")
    p.add_argument("--min-mapq", type=int, default=0, help="skip alignments with lower MAPQ (default 0)")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="also count alignments flagged as duplicates (skipped by default)")
    p.add_argument("--stranded", action="store_true",
                   help="count plus/minus strand separately (53/5/3 modes only): bed mode gives 2 channels "
                        "(plus, minus), genome mode writes <name>_plus.bw and <name>_minus.bw")
    p.add_argument("--fetch-pad", type=int, default=2000,
                   help="entire/entirefraction modes without --maxlen: fragments whose left mate starts more "
                        "than this far left of a window are missed (default 2000)")
    p.add_argument("--shift", default="4,-4", metavar="LEFT,RIGHT",
                   help="shift applied to left and right end (default 4,-4 for Tn5)")
    p.add_argument("--blacklist", help="bed file (named format unless --standard-bed); fragments with an end in it are skipped")
    p.add_argument("--meancounts", type=int, metavar="N",
                   help="(bed mode) mean-pool in bins of N bp (must divide the window length); "
                        "0 averages the whole window")
    p.add_argument("--cpm", action="store_true", help="(bigWig mode) scale to counts per million mapped reads")
    p.add_argument("--chroms", nargs="+", help="(bigWig mode) restrict to these chromosomes")
    p.add_argument("--tile-size", type=int, default=5_000_000, help="(bigWig mode) tile size in bp")
    p.add_argument("--batch-size", type=int, default=500, help="(bed mode) regions per task")
    p.add_argument("-p", "--processes", type=int, default=os.cpu_count() or 1)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    try:
        args.shift = tuple(int(x) for x in args.shift.split(","))
        assert len(args.shift) == 2
    except (ValueError, AssertionError):
        p.error("--shift must be two comma-separated integers, e.g. 4,-4")
    if args.stranded and args.countmode not in SPARSE_MODES:
        p.error("--stranded is only available for countmodes 53, 5 and 3")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    base = args.outname
    if base is None:
        parts = [os.path.splitext(os.path.basename(args.bam))[0]]
        if args.bed:
            parts.insert(0, os.path.splitext(os.path.basename(args.bed))[0] + "_in")
        parts += [args.readmode, args.countmode]
        parts.append("sh" + "_".join(map(str, args.shift)))
        if args.maxlen:
            parts.append(f"max{args.maxlen}")
        base = "_".join(parts)
    outname = os.path.join(args.outdir, base)

    blacklist = None
    if args.blacklist:
        bl = read_bed(args.blacklist, named=not args.standard_bed)
        blacklist = merge_intervals(bl["contigs"], bl["positions"])

    if args.readmode == "fragment" and not check_paired(args.bam):
        sys.exit("No properly paired reads found in the first reads of the BAM; "
                 "for single-end data use --readmode read")

    # Window padding for fetching. Insertion modes find each end through its own
    # mate, so only the shift matters; whole-fragment modes need to see the left
    # mate of fragments that start left of the window.
    if args.countmode in SPARSE_MODES:
        pad = max(abs(s) for s in args.shift) + 10
    else:
        pad = (args.maxlen + 1) if args.maxlen else args.fetch_pad
    opts = {
        "readmode": args.readmode,
        "countmode": args.countmode,
        "shift": args.shift,
        "maxlen": args.maxlen,
        "min_mapq": args.min_mapq,
        "keep_duplicates": args.keep_duplicates,
        "stranded": args.stranded,
        "pad": pad,
    }
    if args.bed:
        run_bed(args, opts, blacklist, outname)
    else:
        run_genome(args, opts, blacklist, outname)
    logger.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
