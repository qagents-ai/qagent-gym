# Build and host private guides

The guides can be built as a complete static Material for MkDocs site. The build
is local and does not create a server, deploy files, change repository visibility,
or grant anyone access. The intended audience remains private and is controlled
by the owner when choosing a hosting destination.

## Build a local site

From a checkout, install the optional documentation tools if they are not already
available. The verified versions are MkDocs **1.6.1** and Material **9.8.1**:

```console
python -m pip install "mkdocs>=1.6.1,<2" "mkdocs-material>=9.8.1,<10"
python scripts/build_guides.py --output .work/private-guides
```

For links and sitemap metadata belonging to a chosen private hosting location:

```console
python scripts/build_guides.py --output .work/private-guides-next --site-url https://your-private-host.example/gym/
```

`--site-url` supplies canonical URL metadata only. It does not contact that host,
publish anything or add authentication. Leave it out until the host is known.
The default theme disables remote font loading; its scripts, styles, search data
and project images are included in the generated site.

The builder copies the repository README to `index.md`, preserves `docs/` and
`assets/`, and includes the other root Markdown documents, such as the specification
and changelog. Relative paths remain intact; README links are redirected to the
generated home page. Navigation is generated from each document's title, grouped
into guides, project references and artwork. New guides, including nested ones,
are discovered automatically at build time.

Builds use strict link/navigation validation. Broken local Markdown links fail the
build rather than silently publishing missing pages. MkDocs runs in a fresh
`.work/guides-build-*` staging tree and the final site is copied to the selected
output only after a successful build. No repository source file is changed.

## Output ownership and retries

The output path must be new or empty. An exclusive reservation prevents another
build from using that path, and every published file is created exclusively.
Neither existing sites nor files appearing during a build are overwritten. The
builder does not delete a filesystem tree. MkDocs' normal cleanup applies only
to the new internal staging site, never the requested final output.

The reservation and any partial output remain after failure. A completed build
has `build-complete.json`; the reservation alone does not indicate completion.
Use a fresh output path to retry. Staging folders are retained for inspection.
Output inside `docs/`, `assets/` or `.git`, and source/output symlinks or Windows
junctions, are rejected to avoid recursive staging or following an unexpected path.

To preview a successful build only on your computer:

```console
python -m http.server 8000 --bind 127.0.0.1 --directory .work/private-guides
```

Open `http://127.0.0.1:8000/`. This loopback preview has no login and must not be
exposed as the private production host. Stop the server when finished.

## Owner-controlled private hosting

The static output is a deployable artifact, not an access-control system. A URL
that is merely difficult to guess, an unlisted deployment, `robots.txt` or a
private source repository does not make the served files private.

For a Sites deployment, the root owner should select the intended account and
destination, configure authenticated access for the named audience, and verify
that anonymous visitors cannot retrieve the pages **or their assets and search
index**. Use the Sites workflow to inspect the destination's available audience
controls before deploying. If that destination cannot enforce the required private
audience, retain the local artifact and choose a host that can. Do not silently
fall back to a public site or publish a public repository/workflow.

The owner controls the final hosting action and audience. Building the files is
not authorization to make them public. No hosting credentials or access tokens
belong in this site, and static JavaScript must not contain a shared password.
This repository contains no automatic public deployment workflow.

Material 9.8.1 currently emits an upstream lifecycle notice during builds. The
helper does not suppress it. Dependency upgrades and any future theme migration
should be reviewed separately from the private hosting decision.
