# =============================================================================
#  The Martian — advanced bolt-on memory (M4 + Talos inspired)
#  Zero dependencies — pure stdlib Python 3
#  Implements most core math from your document
# =============================================================================

import math
import time
import re
from collections import defaultdict, deque, Counter
from typing import List, Dict, Optional, Tuple

class Martian:
    """
    Advanced in-memory manifold memory system.
    - Lotus-weighted graph traversal
    - Nuance / entropy / stability protection
    - Kind-priority retrieval
    - Talos drift monitoring + nudge
    - Identity anchors
    - Pruning
    """

    def __init__(self,
                 max_rooms: int = 800,
                 sim_threshold: float = 0.25,
                 cluster_threshold: int = 600,
                 history_window: int = 20):
        self.rooms: List[Dict] = []  # list of room dicts
        self.room_id_counter = 0
        self.max_rooms = max_rooms
        self.sim_threshold = sim_threshold
        self.cluster_threshold = cluster_threshold
        self.history_window = history_window

        # Graph: room_id → {neighbor_id: lotus_cost}
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)

        # Access order for LRU-ish eviction
        self.access_order = deque(maxlen=max_rooms)

        # Identity anchors (high-stability personal continuity)
        self.anchors: List[Dict] = []

        # Talos state
        self.recent_texts = deque(maxlen=history_window)
        self.attractors: List[str] = []  # user-defined stable vibes/goals

        # Constants from your math
        self.LAMBDA_PI = 0.3
        self.MU_RISK = 0.6
        self.EPS = 1e-10
        self.NOVELTY_BOOST = 0.5
        self.NUANCE_BOOST = 0.5
        self.KIND_PRIORITY = {
            "semantic": 1.0,
            "state": 0.9,
            "commitment": 0.8,
            "episodic": 0.5,
            "unknown": 0.3
        }

    # ─── Pseudo-embedding similarity (no real vectors) ──────────────────────────
    def _pseudo_sim(self, a: str, b: str) -> float:
        """Crude n-gram + length-aware similarity (stand-in for cosine)"""
        if not a or not b:
            return 0.0
        a, b = a.lower(), b.lower()
        def ngrams(s, n): return {s[i:i+n] for i in range(len(s)-n+1)}
        ov3 = len(ngrams(a,3) & ngrams(b,3)) / len(ngrams(a,3) | ngrams(b,3)) if len(a)>2 and len(b)>2 else 0
        ov4 = len(ngrams(a,4) & ngrams(b,4)) / len(ngrams(a,4) | ngrams(b,4)) if len(a)>3 and len(b)>3 else 0
        ov = max(ov3, ov4, 0.0)
        len_r = min(len(a), len(b)) / max(len(a), len(b), 1)
        return ov * (0.35 + 0.65 * len_r**1.4)

    # ─── Math from your document ────────────────────────────────────────────────
    def _nuance(self, text: str) -> float:
        words = re.findall(r'[a-z0-9]+', text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        text = text.lower()
        counts = Counter(text)
        total = len(text)
        h = 0.0
        for c in counts:
            p = counts[c] / total
            h -= p * math.log2(p + self.EPS)
        return h

    def _stability(self, novelty: float, nuance: float) -> float:
        boost = self.NOVELTY_BOOST * novelty + self.NUANCE_BOOST * nuance
        return min(1.0, 1.0 + boost)

    def _lotus_cost(self, dist: float, pi_a: float, risk_a: float) -> float:
        pi_term = self.LAMBDA_PI * pi_a
        risk_term = self.MU_RISK * risk_a
        sing = 1 / (1 - risk_a + 1e-5) if risk_a > 0.8 else 0.0
        return dist + pi_term + risk_term + sing

    def _score_room(self, text: str, ts: float, kind: str, pi: float = None, risk: float = None) -> Dict:
        now = time.time()
        age_h = (now - ts) / 3600
        recency = max(0.05, 1 / (1 + age_h * 0.1))

        novelty = self._entropy(text)
        nuance_val = self._nuance(text)
        stability = self._stability(novelty, nuance_val)

        importance = recency * 0.4 + min(1.0, len(text.split()) / 120) * 0.3 + min(1.0, novelty / 5.0) * 0.3

        meta = {
            "importance": round(importance, 3),
            "novelty": round(novelty, 3),
            "nuance": round(nuance_val, 3),
            "stability": round(stability, 3),
            "pi": pi if pi is not None else round(time.time() % 1, 3),  # pseudo-random [0,1]
            "risk": risk if risk is not None else round((time.time() % 1000)/1000 * 0.4, 3),
            "kind": kind,
            "kind_priority": self.KIND_PRIORITY.get(kind, 0.3),
            "ts": ts,
        }
        return meta

    # ─── Core operations ────────────────────────────────────────────────────────
    def add(self,
            text: str,
            kind: str = "unknown",
            metadata: Optional[Dict] = None,
            is_anchor: bool = False,
            attractor: bool = False) -> int:
        if not text.strip():
            return -1

        ts = time.time()
        max_sim = max((self._pseudo_sim(text, r["text"]) for r in self.rooms), default=0.0)
        if max_sim > 0.94:  # near-duplicate skip
            return -1

        rid = self.room_id_counter
        self.room_id_counter += 1

        meta = self._score_room(text, ts, kind)

        room = {
            "id": rid,
            "text": text,
            "meta": meta,
            **(metadata or {})
        }
        self.rooms.append(room)
        self.access_order.append(rid)

        # Build small-world-ish graph (connect top ~7 similar + any high-stability)
        sims = [(self._pseudo_sim(text, other["text"]), other["id"]) for other in self.rooms if other["id"] != rid]
        sims.sort(reverse=True)
        for sim_val, oid in sims[:7]:  # fwd neighbors
            if sim_val >= self.sim_threshold:
                dist = 1.0 - sim_val  # crude dist proxy
                cost = self._lotus_cost(dist, meta["pi"], meta["risk"])
                self.graph[rid][oid] = round(cost, 4)
                self.graph[oid][rid] = round(cost, 4)  # symmetric for simplicity

        if is_anchor:
            self.anchors.append(room)

        if attractor:
            self.attractors.append(text)

        # Prune if over limit
        if len(self.rooms) > self.max_rooms:
            self._prune_lowest()

        # Clustering stub (placeholder — real would average vectors)
        if len(self.rooms) > self.cluster_threshold:
            pass  # TODO: implement nuance binning merge

        return rid

    def _prune_lowest(self):
        if not self.rooms:
            return
        sorted_rooms = sorted(
            self.rooms,
            key=lambda r: r["meta"]["stability"] * r["meta"]["importance"] * (time.time() - r["meta"]["ts"])/86400
        )
        victim_id = sorted_rooms[0]["id"]
        self.rooms = [r for r in self.rooms if r["id"] != victim_id]
        self.graph.pop(victim_id, None)
        for neigh in self.graph.values():
            neigh.pop(victim_id, None)

    def retrieve(self, query: str, top_k: int = 6, min_score: float = 0.20) -> List[Dict]:
        if not self.rooms:
            return []

        scored = []
        now = time.time()
        for room in self.rooms:
            sim = self._pseudo_sim(query, room["text"])
            if sim < min_score:
                continue

            age_days = (now - room["meta"]["ts"]) / 86400 + 1
            recency = 1 / age_days

            # crude consistency proxy: lower if text has negation flips (very naive)
            contradicts = 1 if "not " in room["text"].lower() or "no " in room["text"].lower() else 0
            consistency = 1 - contradicts / (age_days + 1)

            score = (
                0.4 * sim +
                0.2 * room["meta"]["kind_priority"] +
                0.1 * room["meta"]["importance"] +
                0.1 * room["meta"]["stability"] +
                0.1 * recency +
                0.1 * consistency
            )
            scored.append((score, room))

        scored.sort(reverse=True)
        return [r for _, r in scored[:top_k]]

    def get_context(self, query: str, max_chars: int = 2400) -> str:
        parts = []

        if self.anchors:
            parts.append("Identity anchors:")
            for a in self.anchors[-2:]:
                parts.append(f"• {a['text'][:120]}…")

        if self.attractors:
            parts.append("\nCore attractors to preserve:")
            for att in self.attractors[-3:]:
                parts.append(f"• {att}")

        relevant = self.retrieve(query, top_k=8)
        if relevant:
            parts.append("\nRelevant rooms:")
            for r in relevant:
                age_h = int((time.time() - r["meta"]["ts"]) / 3600)
                s = r["meta"]
                line = f"[{age_h}h ago | {r['meta']['kind']}] {r['text'][:160]}… (score:{s['stability']:.2f} imp:{s['importance']:.2f})"
                parts.append(line)

        ctx = "\n".join(parts)
        return ctx[:max_chars] + "…" if len(ctx) > max_chars else ctx

    # ─── Talos control features ─────────────────────────────────────────────────
    def talos_check(self, new_text: str) -> Dict:
        self.recent_texts.append(new_text.lower())
        if len(self.recent_texts) < 5:
            return {"stable": True, "nudge": None}

        words = []
        for t in self.recent_texts:
            words.extend(re.findall(r'[a-z]+', t))

        if not words:
            return {"stable": True}

        cnt = Counter(words)
        total = len(words)
        entropy = -sum((c/total) * math.log2(c/total + self.EPS) for c in cnt.values())

        # crude phase coherence proxy: repetition rate
        repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        coherence = 1 - min(0.9, repeats / (total - 1))

        drift = entropy < 2.8 or coherence < 0.45

        nudge = None
        if drift and self.attractors:
            nudge = f"Pull toward attractor: {self.attractors[-1][:80]}…"

        return {
            "stable": not drift,
            "entropy": round(entropy, 3),
            "coherence_proxy": round(coherence, 3),
            "nudge_suggestion": nudge
        }

    def status(self) -> str:
        return (
            f"Martian v2 status\n"
            f"  rooms: {len(self.rooms)} / {self.max_rooms}   anchors: {len(self.anchors)}\n"
            f"  attractors: {len(self.attractors)}   graph edges: {sum(len(v) for v in self.graph.values()) // 2}\n"
            f"  next id: {self.room_id_counter}"
        )


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = Martian(max_rooms=120)

    m.add("Julian in Charlotte NC working on infinite memory + epic music.", kind="commitment", is_anchor=True)
    m.add("Chaotic creative energy craving phase-stable flow.", kind="state", attractor=True)
    m.add("Billie Chihiro: 5+ min desperate epic build.", kind="semantic")
    m.add("Blehhhhhh brain pudding mode today.", kind="episodic")

    print(m.status())
    print("\nTalos:", m.talos_check("Want stable creative momentum tonight."))

    print("\nContext for 'music project':")
    print(m.get_context("ideas for Billie Chihiro track"))
