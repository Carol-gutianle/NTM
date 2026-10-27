"""Synthetic alignment, scaling and endpoint checks (no model files)."""
import unittest
import numpy as np
from fuse_bert_tfidf import align_scores, normalized_scores


class FusionTests(unittest.TestCase):
    def test_normalization(self):
        scores = np.array([[1., 3., 2.], [0., 0., 0.]])
        p = normalized_scores(scores)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_allclose(p, normalized_scores(scores * 100 + 5))
        np.testing.assert_array_equal(scores.argmax(1), p.argmax(1))
        with self.assertRaises(ValueError):
            normalized_scores(np.array([[np.nan]]))

    def test_alignment(self):
        a = {"01": dict(model="a", prompt_id="p1", prediction="a"),
             "02": dict(model="b", prompt_id="p2", prediction="b")}
        b = {"02": a["02"], "01": a["01"]}
        scores = np.array([[3., 1.], [1., 3.]])
        aligned_a, aligned_b = align_scores(a, b, scores, scores, ["a", "b"], ["b", "a"])
        np.testing.assert_array_equal(aligned_a, aligned_b)
        for alpha in [0., 1.]:
            fused = alpha * normalized_scores(aligned_a) + (1-alpha) * normalized_scores(aligned_b)
            np.testing.assert_array_equal(fused.argmax(1), [0, 1])
        with self.assertRaises(ValueError):
            align_scores(a, b, scores[::-1], scores, ["a", "b"], ["b", "a"])
        with self.assertRaises(ValueError):
            align_scores(a, {}, scores, scores, ["a", "b"], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
