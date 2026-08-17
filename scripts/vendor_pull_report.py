#!/usr/bin/env python3
"""Pull every git repo under vendor/ and write a daily markdown report.

Discovers repos by walking vendor/ for `.git` entries -- both real repos
(`.git/` directories) and nested submodule gitlinks (`.git` files, e.g.
flashinfer/3rdparty/cutlass is a submodule of flashinfer's own history, not
ours; vendor/ itself has no .gitmodules and is fully gitignored by the
parent repo). Any depth, so vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels gets
found the same as a top-level vendor/cutlass.

Per repo, in order:
  1. Skip (report, don't touch) if the working tree is dirty -- never
     clobbers uncommitted local changes.
  2. Skip (report, don't touch) if HEAD is detached -- nothing sane to
     fast-forward onto.
  3. `git pull --ff-only`, bounded by PULL_TIMEOUT_S. Never merges, never
     rebases, never force-resets: if the branch has diverged from origin,
     this fails loudly and the repo is reported as failed, not silently
     mangled.
  4. If HEAD moved, `git log <old>..<new>` for the commit list (hash, date,
     author, subject) and `git diff-tree --name-status` per commit for the
     changed-files list (capped -- some of these repos have enormous single
     commits).

Stdlib only, no project venv required -- this is meant to also run from
cron, where the venv's interpreter and activation may not be on PATH.

Usage:
    scripts/vendor_pull_report.py                 # vendor/ under the repo root
    scripts/vendor_pull_report.py --vendor-dir X   # explicit path
    scripts/vendor_pull_report.py --dry-run        # discover + report only, no pull

Cron (every morning at 07:00, output also captured to a log):
    0 7 * * * cd /home/shooting-brake007/srswti/shooting-brake && \\
        /usr/bin/python3 scripts/vendor_pull_report.py \\
        >> docs/vendor-reports/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

PULL_TIMEOUT_S = 180
MAX_COMMITS_LISTED = 50
MAX_FILES_PER_COMMIT = 30

REPO_ROOT = Path(__file__).resolve().parent.parent
FS, RS = "\x1f", "\x1e"  # unit/record separators -- immune to commit-message quoting


def find_repos(vendor_dir: Path) -> list[Path]:
    """All directories containing a `.git` entry (dir or file), any depth."""
    repos = []
    stack = [vendor_dir]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        if (d / ".git").exists():
            repos.append(d)
            # Do not descend into a repo's own nested submodules as if they
            # were separate vendor/ top-level entries -- they ARE separate
            # repos (their own .git), and get found on their own once the
            # walk reaches them; no special-casing needed, just don't skip
            # descending, since e.g. flashinfer/3rdparty/cutlass is exactly
            # this case and must still be found.
        for e in entries:
            if e.is_dir() and e.name != ".git":
                stack.append(e)
    return sorted(repos)


def run(cmd: list[str], cwd: Path, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )


def git(cwd: Path, *args: str, timeout: float = 30) -> tuple[int, str, str]:
    try:
        r = run(["git", *args], cwd, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError as e:
        return 127, "", str(e)


class RepoResult:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.remote = "?"
        self.branch = "?"
        self.status = "unknown"   # updated | no-change | skipped-dirty | skipped-detached | failed
        self.detail = ""
        self.before = ""
        self.after = ""
        self.commits: list[dict] = []


def commit_files(repo: Path, sha: str) -> list[str]:
    rc, out, _ = git(repo, "diff-tree", "--no-commit-id", "--name-status", "-r", sha)
    if rc != 0:
        return []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        # Could be a genuinely empty commit, or (far more likely for repos
        # this size) a merge commit: diff-tree with no -m/-c shows nothing
        # for merges since there's no single linear parent diff. Distinguish
        # so an empty list isn't misread as "changed nothing".
        rc, parents, _ = git(repo, "log", "-1", "--format=%P", sha)
        if len(parents.split()) > 1:
            return ["(merge commit -- no single-parent diff to show)"]
    return lines


def process_repo(repo: Path, dry_run: bool) -> RepoResult:
    res = RepoResult(repo)

    rc, out, _ = git(repo, "remote", "get-url", "origin")
    res.remote = out.strip() if rc == 0 else "(no origin)"

    rc, out, _ = git(repo, "symbolic-ref", "--short", "-q", "HEAD")
    detached = rc != 0
    res.branch = out.strip() if not detached else "(detached)"

    rc, out, rev_err = git(repo, "rev-parse", "HEAD")
    res.before = out.strip()[:12] if rc == 0 else "?"

    if rc != 0:
        # Not a normal detached submodule pin -- rev-parse itself failed,
        # e.g. a gitlink whose relative gitdir path doesn't resolve (seen
        # in the wild: flashinfer/flashinfer/data/{cccl,cutlass,spdlog} are
        # copies of flashinfer/3rdparty/* with an unadjusted `gitdir:`
        # pointer -- "not a git repository", not a real checkout).
        res.status = "broken"
        res.detail = (rev_err or "git rev-parse HEAD failed").strip().splitlines()[-1]
        res.after = "?"
        return res

    rc, out, _ = git(repo, "status", "--porcelain")
    dirty = bool(out.strip())

    if dirty:
        res.status = "skipped-dirty"
        res.detail = "uncommitted local changes -- not touched"
        res.after = res.before
        return res
    if detached:
        res.status = "skipped-detached"
        res.detail = "detached HEAD -- not touched"
        res.after = res.before
        return res
    if dry_run:
        res.status = "dry-run"
        res.after = res.before
        return res

    rc, out, err = git(repo, "pull", "--ff-only", timeout=PULL_TIMEOUT_S)
    if rc != 0:
        res.status = "failed"
        res.detail = (err or out).strip().splitlines()[-1] if (err or out).strip() else f"exit {rc}"
        res.after = res.before
        return res

    rc, out, _ = git(repo, "rev-parse", "HEAD")
    res.after = out.strip()[:12] if rc == 0 else res.before

    if res.after == res.before:
        res.status = "no-change"
        return res

    res.status = "updated"
    fmt = f"%H{FS}%h{FS}%an{FS}%aI{FS}%s{RS}"
    rc, out, _ = git(repo, "log", f"{res.before}..{res.after}", f"--format={fmt}")
    for rec in filter(None, out.split(RS)):
        parts = rec.split(FS)
        if len(parts) != 5:
            continue
        full, short, author, date, subject = parts
        files = commit_files(repo, full)
        res.commits.append({
            "hash": short, "full": full, "author": author,
            "date": date, "subject": subject, "files": files,
        })
        if len(res.commits) >= MAX_COMMITS_LISTED:
            break
    return res


def render_report(results: list[RepoResult], started: datetime.datetime, elapsed_s: float) -> str:
    updated = [r for r in results if r.status == "updated"]
    no_change = [r for r in results if r.status == "no-change"]
    skipped = [r for r in results if r.status.startswith("skipped")]
    failed = [r for r in results if r.status == "failed"]
    broken = [r for r in results if r.status == "broken"]
    total_commits = sum(len(r.commits) for r in updated)

    lines = [
        f"# Vendor pull report -- {started:%Y-%m-%d}",
        "",
        f"Run: {started:%Y-%m-%d %H:%M:%S %Z} (took {elapsed_s:.1f}s) \u00b7 "
        f"{len(results)} repos \u00b7 **{len(updated)} updated** "
        f"({total_commits} commits) \u00b7 {len(no_change)} unchanged \u00b7 "
        f"{len(skipped)} skipped \u00b7 {len(failed)} failed \u00b7 "
        f"{len(broken)} broken",
        "",
        "| repo | status | branch | commits | before -> after |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        rel = r.path.relative_to(REPO_ROOT)
        n = str(len(r.commits)) if r.commits else "-"
        lines.append(
            f"| `{rel}` | {r.status} | {r.branch} | {n} | "
            f"`{r.before}` -> `{r.after}` |"
        )
    lines.append("")

    if updated:
        lines.append("## Updated")
        lines.append("")
        for r in updated:
            rel = r.path.relative_to(REPO_ROOT)
            lines.append(f"### `{rel}`")
            lines.append(f"remote: {r.remote} \u00b7 branch: {r.branch} \u00b7 "
                         f"`{r.before}` -> `{r.after}` ({len(r.commits)} commit(s))")
            lines.append("")
            for c in r.commits:
                lines.append(f"- **{c['hash']}** {c['subject']}  ")
                lines.append(f"  {c['author']} \u00b7 {c['date']}")
                files = c["files"]
                if files:
                    shown = files[:MAX_FILES_PER_COMMIT]
                    lines.append("  ```")
                    lines.extend(f"  {f}" for f in shown)
                    if len(files) > MAX_FILES_PER_COMMIT:
                        lines.append(f"  ... +{len(files) - MAX_FILES_PER_COMMIT} more files")
                    lines.append("  ```")
            lines.append("")

    if failed:
        lines.append("## Failed")
        lines.append("")
        for r in failed:
            rel = r.path.relative_to(REPO_ROOT)
            lines.append(f"- `{rel}`: {r.detail}")
        lines.append("")

    if broken:
        lines.append("## Broken (not a valid git checkout, left alone)")
        lines.append("")
        for r in broken:
            rel = r.path.relative_to(REPO_ROOT)
            lines.append(f"- `{rel}`: {r.detail}")
        lines.append("")

    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for r in skipped:
            rel = r.path.relative_to(REPO_ROOT)
            lines.append(f"- `{rel}` ({r.status}): {r.detail}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor-dir", type=Path, default=REPO_ROOT / "vendor")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "vendor-reports")
    ap.add_argument("--dry-run", action="store_true", help="discover and report only, no pull")
    args = ap.parse_args()

    if not args.vendor_dir.is_dir():
        print(f"error: {args.vendor_dir} is not a directory", file=sys.stderr)
        return 1

    started = datetime.datetime.now().astimezone()
    t0 = started.timestamp()
    repos = find_repos(args.vendor_dir)
    print(f"found {len(repos)} repos under {args.vendor_dir}")

    results = []
    for i, repo in enumerate(repos, 1):
        rel = repo.relative_to(REPO_ROOT)
        print(f"  [{i}/{len(repos)}] {rel} ...", end=" ", flush=True)
        res = process_repo(repo, args.dry_run)
        print(f"{res.status} ({res.before} -> {res.after})")
        results.append(res)

    elapsed = datetime.datetime.now().astimezone().timestamp() - t0
    report = render_report(results, started, elapsed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{started:%Y-%m-%d}.md"
    out_path.write_text(report)
    print(f"\nreport -> {out_path}")

    failed = [r for r in results if r.status == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
