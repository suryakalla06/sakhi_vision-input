"""
temporal/affective_gru.py

Lightweight temporal emotional modeling using a pure-numpy GRU.

The GRU runs on the 48-dim affective embedding (from features/affective_embedding.py)
and maintains a 32-dim hidden state that represents the running emotional
memory of the interaction. The hidden state drives the temporal metrics
(emotional_momentum, temporal_coherence, hidden_norm).

Valence/arousal/dominance are NOT read from the (untrained) hidden state — a
random projection would not preserve the sign of the affect (a smile could read
as negative valence). Instead they are a temporally-smoothed copy of the
geometry-based affect that features/affective_embedding.py already computes
(valence_geom / arousal_geom / dominance_geom), which is sign-correct by
construction (AU12/AU6 → positive, AU4/AU15 → negative).

The smoothing is gated by the hidden-state momentum: when the emotional signal
is moving fast we track it quickly; when it is stable we resist per-frame noise.
That keeps the "GRU gating" behaviour (fast update on strong change, smooth when
ambiguous) while guaranteeing the output sign matches the expression.

Design: all state in AffectiveGRU. No I/O. Pure numpy. ~0.1ms per frame.
"""

import numpy as np
from geometry.math_utils import clamp

_INPUT_DIM  = 48    # embedding dimension
_HIDDEN_DIM = 32    # GRU hidden state size
_SEED       = 42    # deterministic initialization
_AFFECT_ALPHA_BASE = 0.3   # min EMA rate for V/A/D (rises toward 1.0 with momentum)


class AffectiveGRU:
    """
    Lightweight GRU for temporal affective sequence modeling.

    Weights are analytically initialized (no training required).
    Hidden state persists across frames — reset() clears it.

    Usage:
        gru   = AffectiveGRU()
        state = gru.update(embedding_dict, timestamp_ms)
    """

    def __init__(self, hidden_dim: int = _HIDDEN_DIM):
        self.h_dim = hidden_dim
        self.h     = np.zeros(hidden_dim, dtype=np.float64)   # hidden state

        # Initialize weights with structured random projections
        rng = np.random.default_rng(_SEED)

        # GRU weight matrices (standard notation)
        # Update gate: z = σ(Wz·x + Uz·h + bz)
        # Reset gate:  r = σ(Wr·x + Ur·h + br)
        # Candidate:   h̃ = tanh(Wh·x + Uh·(r⊙h) + bh)
        # New hidden:  h = (1-z)⊙h + z⊙h̃

        scale_w = 0.1
        scale_u = 0.1

        self.Wz = rng.normal(0, scale_w, (hidden_dim, _INPUT_DIM))
        self.Uz = rng.normal(0, scale_u, (hidden_dim, hidden_dim))
        self.bz = np.full(hidden_dim, -1.0)   # bias toward NOT updating (conservative)

        self.Wr = rng.normal(0, scale_w, (hidden_dim, _INPUT_DIM))
        self.Ur = rng.normal(0, scale_u, (hidden_dim, hidden_dim))
        self.br = np.zeros(hidden_dim)

        self.Wh = rng.normal(0, scale_w, (hidden_dim, _INPUT_DIM))
        self.Uh = rng.normal(0, scale_u, (hidden_dim, hidden_dim))
        self.bh = np.zeros(hidden_dim)

        # Readout: project hidden → coherence (a stability scalar). V/A/D are NOT read
        # from here — see module docstring; they come from the geometric affect below.
        self.W_out = rng.normal(0, 0.1, (1, hidden_dim))
        self.b_out = np.array([0.7])   # bias toward high coherence

        # Smoothed affect (sign-correct, geometry-anchored). Neutral start.
        self._v: float = 0.0   # valence   [-1, 1]
        self._a: float = 0.0   # arousal   [ 0, 1]
        self._d: float = 0.5   # dominance [ 0, 1]

        # Tracking
        self._prev_h: np.ndarray = np.zeros(hidden_dim)
        self._frame_count: int   = 0

    def update(self, embedding_dict: dict, timestamp_ms: float) -> dict:
        """
        Run one GRU step.

        Args:
            embedding_dict : dict from features/affective_embedding.extract_affective_embedding()
            timestamp_ms   : current time in ms (for metadata)

        Returns:
            {
              "gru_valence"      : float   # [-1, 1] GRU-smoothed valence
              "gru_arousal"      : float   # [0, 1]
              "gru_dominance"    : float   # [0, 1]
              "temporal_coherence": float  # [0, 1] consistency of emotional signal
              "hidden_norm"      : float   # L2 norm of hidden state (activation level)
              "emotional_momentum": float  # rate of change of hidden state
              "gru_affect_quadrant": str
            }
        """
        x = np.array(embedding_dict.get("embedding", [0.0] * _INPUT_DIM),
                     dtype=np.float64)

        if len(x) != _INPUT_DIM:
            x = np.zeros(_INPUT_DIM, dtype=np.float64)

        prev_h = self.h.copy()

        # ── GRU forward pass ─────────────────────────────────────────────────
        z  = _sigmoid(self.Wz @ x + self.Uz @ self.h + self.bz)   # update gate
        r  = _sigmoid(self.Wr @ x + self.Ur @ self.h + self.br)   # reset gate
        h_tilde = np.tanh(self.Wh @ x + self.Uh @ (r * self.h) + self.bh)
        self.h  = (1.0 - z) * self.h + z * h_tilde                 # new hidden state

        # ── Emotional momentum (hidden state change rate) ─────────────────────
        delta    = float(np.linalg.norm(self.h - prev_h))
        momentum = clamp(delta / 0.5, 0.0, 1.0)   # normalize

        # ── Affect readout: gated EMA of the geometry-based affect ────────────
        # Sign-correct by construction (see module docstring). The smoothing rate
        # rises with hidden-state momentum: track fast on strong change, resist
        # per-frame noise when the signal is stable.
        v_geom = clamp(float(embedding_dict.get("valence_geom",   0.0)), -1.0, 1.0)
        a_geom = clamp(float(embedding_dict.get("arousal_geom",   0.0)),  0.0, 1.0)
        d_geom = clamp(float(embedding_dict.get("dominance_geom", 0.5)),  0.0, 1.0)

        alpha   = _AFFECT_ALPHA_BASE + (1.0 - _AFFECT_ALPHA_BASE) * momentum
        self._v += alpha * (v_geom - self._v)
        self._a += alpha * (a_geom - self._a)
        self._d += alpha * (d_geom - self._d)
        valence   = clamp(self._v, -1.0, 1.0)
        arousal   = clamp(self._a,  0.0, 1.0)
        dominance = clamp(self._d,  0.0, 1.0)

        # ── Coherence (stability of the emotional signal) ─────────────────────
        coherence = _sigmoid_scalar(float((self.W_out @ self.h + self.b_out)[0]))

        # ── Hidden state norm (overall activation level) ──────────────────────
        h_norm = float(np.linalg.norm(self.h)) / np.sqrt(self.h_dim)

        # ── GRU affect quadrant ───────────────────────────────────────────────
        v_label = "HV" if valence  > 0    else "LV"
        a_label = "HA" if arousal  > 0.45 else "LA"
        quadrant = f"{v_label}{a_label}"

        self._prev_h = prev_h
        self._frame_count += 1

        return {
            "gru_valence"        : round(valence,   4),
            "gru_arousal"        : round(arousal,   4),
            "gru_dominance"      : round(dominance, 4),
            "temporal_coherence" : round(coherence, 4),
            "hidden_norm"        : round(h_norm,    4),
            "emotional_momentum" : round(momentum,  4),
            "gru_affect_quadrant": quadrant,
        }

    def reset(self) -> None:
        """Clear hidden state and smoothed affect (call when face is lost)."""
        self.h = np.zeros(self.h_dim, dtype=np.float64)
        self._v, self._a, self._d = 0.0, 0.0, 0.5
        self._frame_count = 0


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _sigmoid_scalar(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-max(-20, min(20, x)))))
