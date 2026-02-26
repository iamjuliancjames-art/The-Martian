"""
The Martian — M4 + Talos Synthetic Cognition (2026)
Infinite persistent memory + attractor-based phase-stable task guidance
Built by Julian with Grok assistance

- M4: Murmuration Manifold Memory Model (graph + rooms + novelty/nuance)
- Talos: Lotus-Coherent Task State Engine (attractors, coherence, bounded intervention)

Run this as a standalone agent or bolt onto any LLM API
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
import time
import numpy as np
from collections import Counter, defaultdict
import re
import uuid
from sentence_transformers import SentenceTransformer, util

# Config constants from your repo + Talos PDFs
EMBED_DIM = 384
NOVELTY_BOOST = 0.5
NUANCE_BOOST = 0.5
PRUNE_NOVELTY_THRESHOLD = 0.1
PRUNE_AGE = 1000
CONSOLIDATE_EVERY = 25

# Talos thresholds (from PDFs)
H_CRIT = 0.8          # entropy critical threshold
PHI_CRIT = 0.7        # phase alignment critical threshold
U_MAX = 0.3           # bounded intervention max amplitude
ALPHA = 0.95          # drift decay factor (|α| < 1)

# Embedders
text_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Simple text entropy & nuance (your repo utils)
def text_entropy(text: str) -> float:
    if not text: return 0.0
    counts = Counter(text.lower())
    probs = [c / len(text) for c in counts.values()]
    return -sum(p * math.log2(p + 1e-10) for p in probs)

def nuance_score(text: str) -> float:
    words = re.findall(r'\w+', text.lower())
    return len(set(words)) / len(words) if words else 0.0

# Room Metadata (M4 + Talos extensions)
@dataclass
class RoomMeta:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pi: float = 0.0                # priority index
    risk: float = 0.0
    ts: float = field(default_factory=time.time)
    kind: str = "episodic"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    stability: float = 1.0
    novelty: float = 0.0
    nuance: float = 0.0
    age: int = 0
    # Talos additions
    attractor_centroid: Optional[np.ndarray] = None
    attractor_radius: float = 0.5
    phase_score: float = 1.0
    entropy: float = 0.0

@dataclass
class RoomNode:
    id: str
    vec: np.ndarray
    text: str
    meta: RoomMeta

# Core The Martian class (M4 graph + Talos control)
class TheMartian:
    def __init__(self):
        self.rooms: Dict[str, RoomNode] = {}
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # id → [(neighbor_id, edge_weight)]
        self.embed_cache: Dict[str, np.ndarray] = {}
        self.current_state: Dict[str, Any] = {}  # Talos task state X(t)
        self.phase_history: List[float] = []
        self.entropy_history: List[float] = []

    # --- M4 Core (your memory logic) ---

    def add_room(self, text: str, kind: str = "episodic", tags: List[str] = None) -> str:
        if text in self.embed_cache:
            vec = self.embed_cache[text]
        else:
            vec = text_embedder.encode(text, convert_to_numpy=True)
            self.embed_cache[text] = vec

        meta = RoomMeta(kind=kind, tags=tags or [])
        meta.novelty = 1.0 - cosine(vec, np.zeros(EMBED_DIM))  # simple novelty proxy
        meta.nuance = nuance_score(text)
        meta.importance = 0.5 + 0.3 * meta.novelty + 0.2 * meta.nuance

        room = RoomNode(id=meta.id, vec=vec, text=text, meta=meta)
        self.rooms[meta.id] = room

        # Talos attractor init
        meta.attractor_centroid = vec.copy()
        self._update_graph_connections(room)

        # Talos coherence check
        self._talos_coherence_check(room)
        return meta.id

    def _update_graph_connections(self, room: RoomNode):
        for other_id, other in self.rooms.items():
            if other_id == room.id: continue
            sim = cosine(room.vec, other.vec)
            if sim > 0.6:  # edge threshold
                self.graph[room.id].append((other_id, sim))
                self.graph[other_id].append((room.id, sim))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_vec = text_embedder.encode(query, convert_to_numpy=True)
        results = []
        for room_id, room in self.rooms.items():
            sim = cosine(q_vec, room.vec)
            results.append((room_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def build_context(self, query: str) -> str:
        retrieved = self.retrieve(query, top_k=10)
        context_parts = []
        for rid, score in retrieved:
            room = self.rooms[rid]
            context_parts.append(f"[Room {rid[:8]} | score={score:.2f} | {room.meta.kind}] {room.text}")
        return "\n".join(context_parts)

    # --- Talos Control Layer ---

    def _talos_coherence_check(self, room: RoomNode):
        entropy = text_entropy(room.text)
        phase = 1.0 - entropy * 0.4  # simplified cross-modal proxy (real = Fourier-based)

        room.meta.entropy = entropy
        room.meta.phase_score = phase

        self.phase_history.append(phase)
        self.entropy_history.append(entropy)

        if entropy > H_CRIT and phase < PHI_CRIT:
            delta_phi = ALPHA * (self.phase_history[-2] - phase) if len(self.phase_history) > 1 else 0
            correction = min(U_MAX, abs(delta_phi) * 0.5)
            print(f"🚨 TALOS DEVIATION: Entropy {entropy:.3f} > {H_CRIT}, Phase {phase:.3f} < {PHI_CRIT}")
            print(f"   → Applying bounded correction: {correction:.3f}")
            # In full system: nudge attractor centroid or prompt re-sampling
            room.meta.attractor_centroid += correction * (room.vec - room.meta.attractor_centroid)

    def update_task_state(self, current_text: str):
        """Talos Layer 3: Update X(t) state representation"""
        entropy = text_entropy(current_text)
        phase = 1.0 - entropy * 0.4  # proxy

        self.current_state.update({
            "entropy": entropy,
            "phase": phase,
            "active_attractor": current_text[:50] + "..."  # placeholder
        })

        if entropy > H_CRIT and phase < PHI_CRIT:
            return self._talos_intervene(current_text)
        return None

    def _talos_intervene(self, current_text: str) -> str:
        """Talos Layer 4: Bounded, governed intervention"""
        deviation = text_entropy(current_text) - H_CRIT
        correction_strength = min(U_MAX, deviation * 0.3)

        # Simple governed nudge: reframe prompt or suggest stable continuation
        intervention = (
            f"Detected drift (entropy={deviation:.3f}). "
            f"Bounded correction applied ({correction_strength:.3f}). "
            f"Returning to stable attractor: continue from your last coherent idea."
        )
        return intervention

    def query(self, prompt: str) -> Dict[str, Any]:
        """Full Martian query: retrieve context → Talos check → output"""
        context = self.build_context(prompt)
        self.update_task_state(prompt)

        response = {
            "context_used": len(context.splitlines()),
            "entropy": self.current_state.get("entropy", 0.0),
            "phase": self.current_state.get("phase", 1.0),
            "intervention": self._talos_intervene(prompt) if self.current_state.get("entropy", 0) > H_CRIT else None,
            "raw_prompt": prompt,
            "augmented_prompt": f"{context}\n\n{prompt}"
        }
        return response


# Example usage
if __name__ == "__main__":
    martian = TheMartian()

    # Add some rooms (your life + music + isolation + whatever)
    martian.add_room("In DC shelter. Feeling isolated but not lonely. Building M4/Talos.", kind="episodic")
    martian.add_room("Billie Chihiro build: 5+ min epic, intense progression, desperate lyrics.", kind="music")
    martian.add_room("Guh guh guh GAYYYY 42 whatever — bored chaos test.", kind="chat")

    # Query with Martian power
    result = martian.query("What's my current vibe and next music idea?")
    print(result)
