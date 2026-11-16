"""Docs stay correct: generated pages are current, nav targets exist, Mermaid labels are safe."""
import os
import re
import subprocess
import sys

import yaml

from conftest import ROOT


def _nav_files(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for x in node:
            yield from _nav_files(x)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _nav_files(v)


def _load_mkdocs():
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda l, s, n: s)  # mkdocs !!python/name
    return yaml.load(open(os.path.join(ROOT, "mkdocs.yml")), Loader=Loader)


def test_generated_docs_are_current():
    r = subprocess.run([sys.executable, "scripts/gen_docs.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_nav_pages_exist_and_every_page_is_in_nav():
    nav = set(_nav_files(_load_mkdocs()["nav"]))
    docs = os.path.join(ROOT, "docs")
    for rel in nav:
        assert os.path.exists(os.path.join(docs, rel)), f"nav entry missing: {rel}"
    pages = {os.path.relpath(os.path.join(d, f), docs) for d, _, fs in os.walk(docs) for f in fs if f.endswith(".md")}
    assert pages - nav == set(), f"pages not in nav: {pages - nav}"


def test_mermaid_edge_labels_are_quoted():
    """GitHub/Mermaid reject unquoted edge labels containing punctuation like '(' (see README history)."""
    files = [os.path.join(ROOT, "README.md")] + [os.path.join(d, f) for d, _, fs in os.walk(os.path.join(ROOT, "docs"))
                                                 for f in fs if f.endswith(".md")]
    bad = []
    for path in files:
        for block in re.findall(r"```mermaid\n(.*?)```", open(path).read(), re.S):
            for label in re.findall(r"\|([^|\n]*)\|", block):
                if re.search(r"[()\[\]{}]", label) and not (label.startswith('"') and label.endswith('"')):
                    bad.append((os.path.relpath(path, ROOT), label))
    assert not bad, bad
