# Writing docs

This site is built with [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/)
from `docs/` and `mkdocs.yml`.

```bash
uv pip install -p .venv -e ".[docs]"
make docs-serve        # live preview at http://127.0.0.1:8000
make docs              # strict build into site/ (fails on broken links or missing nav pages)
```

## Structure

One directory per top-level tab, each with an `index.md` that serves as the section landing page
(`navigation.indexes`). Add a page by creating the file **and** adding it to `nav:` in `mkdocs.yml`.
`tests/test_docs.py` fails if a nav entry points at a missing file.

## Generated pages

Two pages are generated from the code so they can't drift from it:

| page | source |
|---|---|
| `docs/reference/config.md` | every `TrainConfig` field: type, Hydra-composed default, inline comment |
| `docs/environments/prompts.md` | the exact PointMaze prompts, rendered from `PointMazeTextInterface` |

```bash
make docs-gen          # = python scripts/gen_docs.py
```

`tests/test_docs.py` runs `gen_docs.py --check` and fails if either page is stale. So after changing
`TrainConfig`, a Hydra default, or the PointMaze prompt, run `make docs-gen` and commit the result.

## Diagrams and math

- Mermaid: use a ```` ```mermaid ```` fence. Quote any node or edge label that contains parentheses or
  other punctuation (`A -->|"label (x)"| B`), because GitHub and Mermaid reject unquoted ones.
- Math: `\( inline \)` and `\[ display \]` (MathJax via `pymdownx.arithmatex`).

## Deploying

`.github/workflows/docs.yml` builds the site (strict) on every push to `main` and deploys it to
**GitHub Pages**. One-time setup in the repository: *Settings → Pages → Build and deployment →
Source: GitHub Actions*. The site is then served at `https://jesbu1.github.io/vlm-rl/`.
