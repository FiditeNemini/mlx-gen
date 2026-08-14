#!/usr/bin/env bash
set -euo pipefail

# Stage a filtered copy of the documentation for MkDocs. The validation asset
# bundles under docs/assets/validation are hundreds of megabytes of proof
# media; the published site links them on GitHub instead of shipping them.

SITE_DIR="docs-site"
rm -rf "${SITE_DIR}"
mkdir -p "${SITE_DIR}"

rsync -a --exclude "assets/validation" docs/ "${SITE_DIR}/"

cp README.md "${SITE_DIR}/index.md"
cp CHANGELOG.md "${SITE_DIR}/changelog.md"
if [ -f ACKNOWLEDGEMENTS.md ]; then cp ACKNOWLEDGEMENTS.md "${SITE_DIR}/acknowledgements.md"; fi
# The docs index would conflict with the staged root README (index.md).
mv "${SITE_DIR}/README.md" "${SITE_DIR}/documentation-index.md"

python - <<'PY'
from pathlib import Path

site = Path("docs-site")
blob = "https://github.com/lpalbou/mlx-gen/blob/main/docs/assets/validation/"
src_blob = "https://github.com/lpalbou/mlx-gen/blob/main/src/"

for path in site.rglob("*.md"):
    text = path.read_text(encoding="utf-8")
    updated = (
        text.replace("](assets/validation/", f"]({blob}")
        .replace("](./assets/validation/", f"]({blob}")
        .replace("](../assets/validation/", f"]({blob}")
        .replace("](../../assets/validation/", f"]({blob}")
    )
    updated = updated.replace("](../../src/", f"]({src_blob}").replace("](../src/", f"]({src_blob}")
    # Root README and CHANGELOG link into docs/ directly; retarget when staged.
    if path.name in ("index.md", "changelog.md"):
        updated = updated.replace("](docs/", "](")
    if path.name == "index.md":
        updated = updated.replace("](ACKNOWLEDGEMENTS.md", "](acknowledgements.md")
    if path.name == "documentation-index.md":
        root_blob = "https://github.com/lpalbou/mlx-gen/blob/main/"
        updated = (
            updated.replace("](../README.md", "](index.md")
            .replace("](../CHANGELOG.md", "](changelog.md")
            .replace("](../ACKNOWLEDGEMENTS.md", "](acknowledgements.md")
            .replace("](../CONTRIBUTING.md", f"]({root_blob}CONTRIBUTING.md")
            .replace("](../SECURITY.md", f"]({root_blob}SECURITY.md")
            .replace("](../CODE_OF_CONDUCT.md", f"]({root_blob}CODE_OF_CONDUCT.md")
        )
    if updated != text:
        path.write_text(updated, encoding="utf-8")
PY

echo "staged $(find "${SITE_DIR}" -name '*.md' | wc -l | tr -d ' ') markdown pages into ${SITE_DIR}/"
