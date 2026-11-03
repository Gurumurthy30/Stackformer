# Organization Name References (`stackformer-labs`)

This document lists every file in the repository that contains references to the GitHub organization `stackformer-labs`. When migrating or renaming the organization or repository name, update the references in these files.

---

## Source and Configuration Files

1. **`README.md`**:
   - Badge image URLs and links (CI status, License, Repository link, Discussion, Issues, Releases)
   - Git clone instructions
   - Citation BibTeX entry (`howpublished` URL)
2. **`pyproject.toml`**:
   - `[project.urls]`: `Repository`, `Releases`, `Issue-Tracker`, `Discussions`, `Documentation`
3. **`Dockerfile`**:
   - Label annotations or repository git clone URLs
4. **`mkdocs.yml`**:
   - `repo_url`: `https://github.com/stackformer-labs/Stackformer`

---

## Documentation Source Files

1. **`docs/index.md`**:
   - Repository links and community badge references
2. **`docs/developer-guide/index.md`**:
   - GitHub contribution links and issue tracker URLs
3. **`docs/getting-started/installation.md`**:
   - Source installation command (`git clone https://github.com/stackformer-labs/Stackformer.git`)

---

## Generated Documentation (Auto-rebuilt on deployment)

- `site/*` (Generated MkDocs HTML and search index files; rebuilt automatically when `mkdocs build` is executed).
