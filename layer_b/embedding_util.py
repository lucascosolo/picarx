#!/usr/bin/env python3
# /home/picarx/layer_b/embedding_util.py
"""
Optional text-embedding helper for semantic situation matching in
coach.py.

The coach keys everything on an exact-string situation_key
("novel_object:sofa"), which means "novel_object:couch" or a brand-new
label starts learning from scratch even though the robot already knows
a perfectly good maneuver for the near-identical "sofa". This module
turns a short situation description into a vector so the coach can find
the NEAREST situation it already has experience with and transfer that
experience, instead of only matching identical strings.

It is ENTIRELY OPTIONAL and fail-soft: if onnxruntime / tokenizers /
numpy aren't installed, or the model files aren't present, `available`
stays False and every caller silently falls back to the existing
exact-string behavior. The robot works exactly as before without it -
this only ever ADDS generalization when the model is set up. See
SETUP_embeddings.md for the one-time install/download steps.

Model: sentence-transformers/all-MiniLM-L6-v2 exported to ONNX (384-dim,
~90MB). Encoding one short situation string is a few milliseconds on a
Pi 4 and only happens on a genuinely new situation, so it adds no
steady-state load.
"""
import os
import robot_config

EMBED_MODEL_PATH = str(robot_config.get(
    "embeddings", "model_path",
    "/home/picarx/layer_b/data/models/minilm/model.onnx", env="EMBED_MODEL_PATH"))
EMBED_TOKENIZER_PATH = str(robot_config.get(
    "embeddings", "tokenizer_path",
    "/home/picarx/layer_b/data/models/minilm/tokenizer.json", env="EMBED_TOKENIZER_PATH"))


class Embedder:
    def __init__(self):
        self.available = False
        self._session = None
        self._tokenizer = None
        self._np = None
        self._input_names = set()

        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            print(f"Embedder: disabled (missing dependency: {e}). "
                  f"pip install onnxruntime tokenizers numpy to enable "
                  f"semantic situation matching.")
            return

        if not (os.path.exists(EMBED_MODEL_PATH) and os.path.exists(EMBED_TOKENIZER_PATH)):
            print(f"Embedder: disabled (model/tokenizer not found at "
                  f"{EMBED_MODEL_PATH} / {EMBED_TOKENIZER_PATH}). See SETUP_embeddings.md.")
            return

        try:
            self._np = np
            self._tokenizer = Tokenizer.from_file(EMBED_TOKENIZER_PATH)
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1   # embedding is tiny and rare; leave cores for vision/audio
            self._session = ort.InferenceSession(
                EMBED_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
            self._input_names = {i.name for i in self._session.get_inputs()}
            self.available = True
            print("Embedder: semantic situation matching enabled (MiniLM ONNX).")
        except Exception as e:
            print(f"Embedder: failed to initialize, disabled: {e}")

    def encode(self, text):
        """Return an L2-normalized 384-float list, or None if unavailable."""
        if not self.available:
            return None
        try:
            np = self._np
            enc = self._tokenizer.encode(text or "")
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            candidate = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            }
            inputs = {k: v for k, v in candidate.items() if k in self._input_names}
            out = self._session.run(None, inputs)[0]      # [1, seq, hidden]
            hidden = out[0]                                # [seq, hidden]
            m = mask[0][:, None]                           # [seq, 1]
            summed = (hidden * m).sum(axis=0)
            counts = np.clip(m.sum(axis=0), 1e-9, None)
            vec = summed / counts                          # mean pooling
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32).tolist()
        except Exception as e:
            print(f"Embedder: encode failed: {e}")
            return None

    @staticmethod
    def cosine(a, b):
        """Cosine similarity of two encode() outputs. Because encode()
        L2-normalizes, this is just the dot product. Range [-1, 1]."""
        if not a or not b or len(a) != len(b):
            return -1.0
        return sum(x * y for x, y in zip(a, b))
