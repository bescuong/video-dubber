"""Developer Publisher Script: Pushes source tree & HotPatch Release to GitHub (bescuong/video-dubber).

Reads GitHub PAT strictly from %APPDATA%/VideoDubber/admin_github_secret.json or GITHUB_TOKEN env var
so no secret is ever stored in the repository.
"""
from __future__ import annotations

import base64
import http.client
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "bescuong/video-dubber"
APP_DATA_SECRET = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "VideoDubber" / "admin_github_secret.json"

TRACKED_FILES = [
    ".gitignore",
    "app.py",
    "cli.py",
    "VideoDubber.spec",
    "installer.iss",
    "Chay_ung_dung.bat",
    "Dong_goi_ung_dung.bat",
    "Day_code_len_GitHub.bat",
    "HD_CAU_HINH_FIREBASE.md",
    "firebase_config.json.example",
    "version.json",
    "icon.ico",
    "tools/push_to_github.py",
]


def _get_token() -> str:
    env_tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if env_tok:
        return env_tok
    if APP_DATA_SECRET.is_file():
        return json.loads(APP_DATA_SECRET.read_text(encoding="utf-8-sig")).get("token", "").strip()
    raise RuntimeError(f"Missing GitHub token in {APP_DATA_SECRET} or GITHUB_TOKEN env var.")


def _gh_api(token: str, path_or_url: str, method: str = "GET", payload: dict | None = None):
    url = path_or_url if path_or_url.startswith("http") else f"https://api.github.com{path_or_url}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "VideoDubber-Publisher",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def ensure_repo_exists(token: str):
    _, repo_name = REPO.split("/", 1)
    try:
        info = _gh_api(token, f"/repos/{REPO}")
        print(f"[GitHub] Repo exists: {info['html_url']}")
        return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    print(f"[GitHub] Creating public repository {REPO}...")
    info = _gh_api(
        token,
        "/user/repos",
        method="POST",
        payload={
            "name": repo_name,
            "description": "Video Dubber Studio — AI Video Dubbing, Multi-Character TTS & Smart Subtitle Replacement",
            "private": False,
            "auto_init": True,
        },
    )
    time.sleep(2.0)
    print(f"[GitHub] Created repository: {info['html_url']}")


def push_source_tree(token: str, commit_msg: str):
    # Collect tracked files + dubber/*.py
    rel_paths = list(TRACKED_FILES)
    for py_f in sorted((ROOT / "dubber").glob("*.py")):
        rel_paths.append(f"dubber/{py_f.name}")

    # Get latest commit on main
    ref = _gh_api(token, f"/repos/{REPO}/git/ref/heads/main")
    parent_sha = ref["object"]["sha"]
    parent_commit = _gh_api(token, f"/repos/{REPO}/git/commits/{parent_sha}")
    base_tree_sha = parent_commit["tree"]["sha"]

    tree_items = []
    for rel in rel_paths:
        fp = ROOT / rel
        if not fp.is_file():
            continue
        raw_bytes = fp.read_bytes()
        blob = _gh_api(
            token,
            f"/repos/{REPO}/git/blobs",
            method="POST",
            payload={
                "content": base64.b64encode(raw_bytes).decode("ascii"),
                "encoding": "base64",
            },
        )
        tree_items.append({
            "path": rel,
            "mode": "100644",
            "type": "blob",
            "sha": blob["sha"],
        })

    new_tree = _gh_api(
        token,
        f"/repos/{REPO}/git/trees",
        method="POST",
        payload={"base_tree": base_tree_sha, "tree": tree_items},
    )
    new_commit = _gh_api(
        token,
        f"/repos/{REPO}/git/commits",
        method="POST",
        payload={
            "message": commit_msg,
            "tree": new_tree["sha"],
            "parents": [parent_sha],
        },
    )
    _gh_api(
        token,
        f"/repos/{REPO}/git/refs/heads/main",
        method="PATCH",
        payload={"sha": new_commit["sha"], "force": False},
    )
    print(f"[GitHub] Pushed {len(tree_items)} source files -> commit {new_commit['sha'][:8]}")


def build_hotpatch_zip(version: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(ROOT / "app.py", "app.py")
        zf.write(ROOT / "cli.py", "cli.py")
        for py_f in sorted((ROOT / "dubber").glob("*.py")):
            zf.write(py_f, f"dubber/{py_f.name}")
        zf.writestr(
            "patch_manifest.json",
            json.dumps({"version": version, "built_at": time.time()}, indent=2),
        )
    return buf.getvalue()


def publish_release(token: str, version: str, notes: str):
    tag = f"v{version}"
    try:
        rel = _gh_api(
            token,
            f"/repos/{REPO}/releases",
            method="POST",
            payload={
                "tag_name": tag,
                "target_commitish": "main",
                "name": f"Video Dubber Studio v{version}",
                "body": notes,
                "draft": False,
                "prerelease": False,
            },
        )
    except urllib.error.HTTPError as e:
        if e.code != 422:
            raise
        rel = _gh_api(token, f"/repos/{REPO}/releases/tags/{tag}")
        rel = _gh_api(
            token,
            f"/repos/{REPO}/releases/{rel['id']}",
            method="PATCH",
            payload={"name": f"Video Dubber Studio v{version}", "body": notes},
        )

    asset_name = f"VideoDubber_HotPatch_v{version}.zip"
    for a in rel.get("assets", []):
        if a.get("name") == asset_name:
            _gh_api(token, f"/repos/{REPO}/releases/assets/{a['id']}", method="DELETE")

    zip_bytes = build_hotpatch_zip(version)
    upload_base = rel["upload_url"].split("{")[0]
    up_url = f"{upload_base}?name={urllib.parse.quote(asset_name)}"
    parsed = urllib.parse.urlparse(up_url)

    conn = http.client.HTTPSConnection(parsed.netloc, timeout=60)
    conn.putrequest("POST", f"{parsed.path}?{parsed.query}")
    conn.putheader("Authorization", f"Bearer {token}")
    conn.putheader("User-Agent", "VideoDubber-Publisher")
    conn.putheader("Accept", "application/vnd.github+json")
    conn.putheader("Content-Type", "application/zip")
    conn.putheader("Content-Length", str(len(zip_bytes)))
    conn.endheaders()
    conn.send(zip_bytes)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", errors="ignore")
    conn.close()
    if resp.status not in (200, 201):
        raise RuntimeError(f"Asset upload failed ({resp.status}): {body[:300]}")

    up_data = json.loads(body)
    print(f"[GitHub] Published Release {tag}: {rel['html_url']}")
    print(f"[GitHub] HotPatch Asset ({len(zip_bytes)//1024} KB): {up_data['browser_download_url']}")
    return rel["html_url"], up_data["browser_download_url"]


def main():
    vdata = json.loads((ROOT / "version.json").read_text(encoding="utf-8"))
    ver = str(vdata["version"]).lstrip("vV")
    notes = vdata.get("notes", f"Release v{ver}")
    token = _get_token()

    ensure_repo_exists(token)
    push_source_tree(token, f"release: Video Dubber Studio v{ver}")
    rel_url, patch_url = publish_release(token, ver, notes)

    # Update version.json with hotpatch_url and push once more cleanly
    vdata["hotpatch_url"] = patch_url
    (ROOT / "version.json").write_text(json.dumps(vdata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    push_source_tree(token, f"chore: update version.json hotpatch url for v{ver}")
    print(f"[DONE] Repository & Release live at: {rel_url}")


if __name__ == "__main__":
    main()
