import ast
import json
from pathlib import Path
import tempfile
import unittest

import reproduce_decoder as decoder


class DecoderTests(unittest.TestCase):
    def test_extract_first_literal_ignores_later_reassignment(self):
        source = "X = 'hello'\nX = X.replace('h', 'H')\n"
        self.assertEqual(decoder.literal_assignment(source, "X"), "hello")

    def test_remaps_embedded_source_too(self):
        inner = "root = '/kaggle/input'\n"
        source = f"root = '/kaggle/input'\nembedded = {inner!r}\n"
        mapped = decoder.remap_source(source, Path("/tmp/decoder-input"))
        self.assertNotIn("/kaggle/input", mapped)
        compile(mapped, "outer", "exec")
        embedded = decoder.literal_assignment(mapped, "embedded")
        self.assertEqual(embedded, "root = '/tmp/decoder-input'\n")
        compile(embedded, "inner", "exec")

    def test_reject_unsafe_path(self):
        with self.assertRaises(ValueError):
            decoder.validate_input_path(Path("/tmp/it's-bad"))

    def test_reject_modified_source(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            (bundle / "original.ipynb").write_text("{}")
            with self.assertRaises(ValueError):
                decoder.load_notebook(bundle)

    def test_inventory_complete_and_incomplete_lora(self):
        nb = {"cells": [{}, {}, {"source": (
            "_required = {'ATLAS bundle': '**/atlas_bundle.json'}\n"
            "_lora24_files = ('adapter/adapter_model.safetensors', 'heads.safetensors')\n"
        )}]}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ck = root / "one/lora24/final"
            (ck / "adapter").mkdir(parents=True)
            (ck / "adapter/adapter_model.safetensors").touch()
            report = decoder.asset_report(nb, root)
            self.assertFalse(report["original LoRA24"]["ok"])
            self.assertEqual(report["original LoRA24"]["incomplete"][str(ck)],
                             ["heads.safetensors"])
            (ck / "heads.safetensors").touch()
            self.assertTrue(decoder.asset_report(nb, root)["original LoRA24"]["ok"])

    @unittest.skipUnless((decoder.DEFAULT_BUNDLE / "original.ipynb").exists(),
                         "Run prepare first for original-source integration test")
    def test_original_all_cells_compile_after_remapping(self):
        nb = decoder.load_notebook(decoder.DEFAULT_BUNDLE)
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = decoder.remap_source(decoder.cell_source(cell), Path("/tmp/decoder"))
            compile(source, "original", "exec")
        source = decoder.remap_source(decoder.cell_source(nb["cells"][3]),
                                      Path("/tmp/decoder"))
        embedded = decoder.literal_assignment(source, "_GAUSSIAN_SOURCE")
        self.assertNotIn("/kaggle/input", embedded)
        ast.parse(embedded)


if __name__ == "__main__":
    unittest.main()
