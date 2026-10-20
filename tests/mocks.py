"""Model-free stand-ins. They exercise plumbing only; their outputs say nothing about methods."""
import numpy as np

from ccpilot.textutil import word_tokens


def lexical_score_fn(tok):
    """Pseudo 'attention': 1 for tokens whose word occurs in the last 64 tokens (question window)."""
    def fn(ids):
        tail = set(word_tokens(tok.decode(ids[-64:])))
        rng = np.random.default_rng(len(ids))
        s = rng.random(len(ids)) * 0.01
        for i, t in enumerate(ids):
            w = word_tokens(tok.decode([t]))
            if w and w[0] in tail:
                s[i] += 1.0
        return s
    return fn


def mock_llmlingua_fn(context, target_token, word_sep="\t\t|\t\t"):
    """Keeps every k-th word so that ~target_token words remain; returns LLMLingua-2-like labels."""
    words = context.split()
    k = max(1, round(len(words) / max(1, target_token / 1.3)))
    keep = [i % k == 0 for i in range(len(words))]
    text = " ".join(w for w, m in zip(words, keep) if m)
    labeled = word_sep.join(f"{w} {int(m)}" for w, m in zip(words, keep))
    return text, labeled
