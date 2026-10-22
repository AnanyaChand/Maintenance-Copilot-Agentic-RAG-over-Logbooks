"""Problem-only mutual-neighbour graph and conservative complete-link patterns."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json

import faiss
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .pattern_fingerprints import fingerprint, conflicts, meaningful

FRAMING = "Repeated historical issue pattern across the MaintNet corpus"
DISCLAIMER = ("Patterns are similarity-based groupings of historical logbook text. "
              "They do not establish aircraft-specific recurrence, cause, or approved maintenance action.")
ACTION_HEADING = "Historical action variants recorded"
ACTION_WARNING = "Multiple historical action variants were recorded. No single action is endorsed."


@dataclass(frozen=True)
class PatternConfig:
    neighbours: int = 20
    cosine: float = .84
    distance: float = .16
    minimum_size: int = 3
    mean_cohesion: float = .86
    minimum_cohesion: float = .80
    stability: float = .80
    perturbation: float = .02
    location_boost: float = .02
    identifier_boost: float = .03

    def __post_init__(self):
        if self.minimum_size < 3 or self.neighbours < 1:
            raise ValueError("Patterns require at least three records and positive neighbours")
        for name, value in asdict(self).items():
            if name not in {"minimum_size", "neighbours"} and not 0 <= value <= 1:
                raise ValueError(name)


def components(n, pairs):
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        parent[root(b)] = root(a)
    groups = defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)
    return list(groups.values())


def action_variants(records):
    # Equality of normalized action text is an intentionally strict, transparent
    # similarity rule. It never collapses different instructions or outcomes.
    groups = defaultdict(list)
    for record in records:
        groups[record.action_normalized].append(record)
    return [dict(record_ids=[r.record_id for r in rows], count=len(rows),
                 representative_original=rows[0].action_original,
                 original_actions=sorted({r.action_original for r in rows}))
            for _, rows in sorted(groups.items())]


def analyze(records, vectors, config=PatternConfig()):
    """Vectors must correspond to records, and be encoded from PROBLEM only."""
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("duplicate record IDs")
    v = np.asarray(vectors, dtype=np.float32).copy()
    if v.ndim != 2 or len(v) != len(records) or not np.isfinite(v).all() or np.any(np.linalg.norm(v, axis=1) == 0):
        raise ValueError("invalid problem vectors")
    faiss.normalize_L2(v)
    fps = [fingerprint(r) for r in records]
    # Duplicate prototypes avoid top-k saturation by repeated identical text.
    units = defaultdict(list)
    for i, (r, fp) in enumerate(zip(records, fps)):
        signature = (r.problem_normalized, tuple(fp.locations), tuple(fp.technical_identifiers),
                     tuple(fp.negated_symptoms), tuple(fp.states))
        if conflicts(fp, fp) or not meaningful(fp):
            signature += (r.record_id,)
        units[signature].append(i)
    members = sorted(units.values(), key=lambda ids: records[ids[0]].record_id)
    reps = [ids[0] for ids in members]
    u = v[reps]
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(v.shape[1]); index.add(u)
    _, neighbours = index.search(u, min(len(u), config.neighbours + 1))
    ranks = [{int(j): rank for rank, j in enumerate([j for j in row if j != i], 1)}
             for i, row in enumerate(neighbours)]
    candidates = []
    for i in range(len(u)):
        for j in sorted(ranks[i]):
            if j <= i or i not in ranks[j]:
                continue
            a, b = fps[reps[i]], fps[reps[j]]
            shared = sorted(meaningful(a) & meaningful(b))
            if conflicts(a, b) or not shared:
                continue
            cosine = float(np.clip(u[i] @ u[j], -1, 1))
            candidates.append((i, j, cosine, shared))

    def pair_score(i, j):
        a, b = fps[reps[i]], fps[reps[j]]
        cos = float(np.clip(u[i] @ u[j], -1, 1))
        eligible = not conflicts(a, b) and bool(meaningful(a) & meaningful(b))
        boosted = min(1., cos + config.location_boost * bool(set(a.locations) & set(b.locations))
                      + config.identifier_boost * bool(set(a.technical_identifiers) & set(b.technical_identifiers)))
        return cos, boosted, eligible

    def cluster(threshold, distance, method):
        blocks = components(len(u), [(i, j) for i, j, cos, _ in candidates if cos >= threshold])
        if method == "connected_components":
            return [set(k for i in block for k in members[i]) for block in blocks]
        result = []
        for block in blocks:
            if len(block) == 1:
                result.append(set(members[block[0]])); continue
            distances = np.zeros((len(block), len(block)))
            for x in range(len(block)):
                for y in range(x):
                    cos, score, eligible = pair_score(block[x], block[y])
                    distances[x, y] = distances[y, x] = max(0, 1-score) if eligible and cos >= threshold else 1.
            labels = fcluster(linkage(squareform(distances, checks=False), method="complete"), distance, criterion="distance")
            grouped = defaultdict(set)
            for i, label in zip(block, labels):
                grouped[int(label)].update(members[i])
            result.extend(grouped.values())
        return result

    frequencies = Counter(t for fp in fps for t in meaningful(fp) | set(fp.subsystems) | set(fp.states))
    def summarize(ids):
        ordered = sorted(ids, key=lambda i: records[i].record_id)
        shared = {key: sorted(set.intersection(*(set(getattr(fps[i], key)) for i in ordered)))
                  for key in ("components", "subsystems", "symptoms", "states", "locations", "technical_identifiers")}
        # Weighted prototype similarities avoid large duplicate pair matrices.
        groups = [([k for k in m if k in ids], r) for m, r in zip(members, reps) if m[0] in ids]
        total, minimum, count, issues = 0., 1., 0, set()
        for x, (a, ra) in enumerate(groups):
            n = len(a)*(len(a)-1)//2
            total += n; count += n
            issues.update(conflicts(fps[ra], fps[ra]))
            for b, rb in groups[:x]:
                cos = float(np.clip(v[ra] @ v[rb], -1, 1)); n = len(a)*len(b)
                total += n*cos; count += n; minimum = min(minimum, cos)
                issues.update(conflicts(fps[ra], fps[rb]))
        mean = total/count if count else 1.
        component = shared["components"] or shared["subsystems"]
        symptom = shared["symptoms"] or shared["states"]
        reasons = []
        if len(ids) < config.minimum_size: reasons.append("below minimum size")
        if mean < config.mean_cohesion or minimum < config.minimum_cohesion: reasons.append("below cohesion threshold")
        if not component or not symptom: reasons.append("no shared component/subsystem and symptom/state")
        if issues: reasons.append("critical fingerprint conflict")
        pick = lambda terms: min(terms, key=lambda t: (frequencies[t], t)) if terms else ""
        label = f"Repeated {pick(component)} {pick(symptom)} reports"
        record_ids = [records[i].record_id for i in ordered]
        center = v[ordered].mean(axis=0)
        representative_ids = sorted(ordered, key=lambda i: (-float(v[i] @ center), records[i].record_id))[:3]
        return dict(pattern_id="PAT-"+hashlib.sha256("\n".join(record_ids).encode()).hexdigest()[:12],
                    label=label, record_ids=record_ids, size=len(ids), mean_similarity=mean,
                    minimum_similarity=minimum, shared=shared, conflicts=sorted(issues),
                    representatives=[dict(record_id=records[i].record_id, problem_original=records[i].problem_original) for i in representative_ids],
                    exclusion_reasons=reasons)

    audits = {}
    for method in ("connected_components", "complete_link"):
        base = cluster(config.cosine, config.distance, method)
        variants = [cluster(c, d, method) for c, d in (
            (config.cosine-config.perturbation, config.distance), (config.cosine+config.perturbation, config.distance),
            (config.cosine, config.distance-config.perturbation), (config.cosine, config.distance+config.perturbation))]
        rows = []
        for ids in base:
            if len(ids) < config.minimum_size:
                continue
            row = summarize(ids)
            row["stability"] = min(max(len(ids & other)/len(ids | other) for other in version) for version in variants)
            if row["stability"] < config.stability: row["exclusion_reasons"].append("unstable membership")
            rows.append(row)
        audits[method] = rows
    patterns = []
    for row in audits["complete_link"]:
        if not row["exclusion_reasons"]:
            variants = action_variants([records[i] for i in range(len(records)) if records[i].record_id in set(row["record_ids"])])
            patterns.append(dict(row, action_heading=ACTION_HEADING, action_variants=variants,
                                 action_warning=ACTION_WARNING if len(variants)>1 else None))
    edges = []
    for m in members:
        for j in m[1:]:
            edges.append(dict(source=records[m[0]].record_id, target=records[j].record_id, cosine=1.,
                              weight=1., kind="identical_problem_star", shared_terms=sorted(meaningful(fps[j]))))
    for i, j, cos, shared in candidates:
        if cos >= config.cosine:
            edges.append(dict(source=records[reps[i]].record_id, target=records[reps[j]].record_id,
                              cosine=cos, weight=pair_score(i,j)[1], kind="mutual_prototype_neighbour",
                              shared_terms=shared, shared_locations=sorted(set(fps[reps[i]].locations)&set(fps[reps[j]].locations)),
                              ranks=[ranks[i][j], ranks[j][i]]))
    return dict(patterns=patterns, audits=audits, edges=edges,
                nodes=[fp.model_dump() for fp in fps], prototype_count=len(u), config=asdict(config))
