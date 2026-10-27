import unittest
from ensemble_vote import hard_vote, align, metrics


class VotingTests(unittest.TestCase):
    def test_majority(self):
        self.assertEqual(hard_vote({"a": "x", "b": "y", "c": "y"}, ["a", "b", "c"]),
                         ("y", 2, False, True))

    def test_tie(self):
        self.assertEqual(hard_vote({"a": "x", "b": "y", "c": "z"}, ["b", "c", "a"]),
                         ("y", 1, True, False))
        # Preferred member is not tied for the lead: skip it.
        self.assertEqual(hard_vote(dict(a="z", b="x", c="y", d="x", e="y"),
                                   list("abcde")), ("x", 2, True, False))

    def test_plurality(self):
        self.assertEqual(hard_vote(dict(a="x", b="x", c="y", d="z", e="w"),
                                   list("abcde")), ("x", 2, False, False))

    def test_alignment_and_metrics(self):
        a = {"01": dict(model="x", prompt_id="p1", prediction="x"),
             "02": dict(model="y", prompt_id="p2", prediction="y")}
        b = {"02": a["02"].copy(), "01": a["01"].copy()}
        self.assertEqual(align({"a": a, "b": b}), (["01", "02"], ["x", "y"]))
        b["01"]["prompt_id"] = "bad"
        with self.assertRaises(ValueError):
            align({"a": a, "b": b})
        m, _ = metrics(["x", "y"], ["x", "x"], ["x", "y"])
        self.assertEqual(m["accuracy"], .5)
        self.assertAlmostEqual(m["macro_f1"], 1/3)
        with self.assertRaises(ValueError):
            hard_vote({"a": "x", "b": "x"}, ["a", "a"])


if __name__ == "__main__":
    unittest.main()
