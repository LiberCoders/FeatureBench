#!/usr/bin/env python3
"""Pre-pull Docker images required by ACE-Bench before running infer.

Usage:
  python acebench/scripts/pull_images.py --mode full
  python acebench/scripts/pull_images.py --mode lite
  python acebench/scripts/pull_images.py --mode /path/to/images.txt

Mode:
  - full: use acebench/resources/constants/full_images.txt
  - lite: use acebench/resources/constants/lite_images.txt
  - custom path: any txt file where each line is an image reference

Notes:
  - Empty lines and lines starting with '#' are ignored.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ALLOWED_REGISTRY_PREFIXES = ("docker.io/", "ghcr.io/", "gcr.io/")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_list_path(mode: str) -> Path:
    root = _repo_root()
    if mode == "full":
        return root / "acebench" / "resources" / "constants" / "full_images.txt"
    if mode == "lite":
        return root / "acebench" / "resources" / "constants" / "lite_images.txt"
    raise ValueError(f"Unknown mode: {mode}")


def _normalize_image_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if "/" not in s or not s.startswith(ALLOWED_REGISTRY_PREFIXES):
        s = "docker.io/" + s
    return s


def _load_images_from_txt(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Image list file not found: {path}")

    images: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        img = _normalize_image_name(line)
        if img:
            images.append(img)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for img in images:
        if img not in seen:
            uniq.append(img)
            seen.add(img)
    return uniq


@dataclass
class PullResult:
    image: str
    ok: bool
    error: str | None = None
    output: str | None = None


def _docker_pull(image: str, *, capture_output: bool = False) -> PullResult:
    try:
        proc = subprocess.run(
            ["docker", "pull", image],
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
        if proc.returncode == 0:
            return PullResult(image=image, ok=True, output=proc.stdout if capture_output else None)
        return PullResult(
            image=image,
            ok=False,
            error=f"docker pull exited with {proc.returncode}",
            output=proc.stdout if capture_output else None,
        )
    except FileNotFoundError:
        return PullResult(image=image, ok=False, error="docker command not found")
    except Exception as e:  # noqa: BLE001
        return PullResult(image=image, ok=False, error=str(e))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-pull Docker images for ACE-Bench")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        help="full | lite | /path/to/images.txt (default: full)",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Number of concurrent docker pulls (default: 1)",
    )

    args = parser.parse_args(argv)
    mode = str(args.mode).strip()
    n_concurrent = int(args.n_concurrent)

    if n_concurrent <= 0:
        print("❌ --n-concurrent must be >= 1")
        return 2

    if shutil.which("docker") is None:
        print("❌ docker not found in PATH. Please install Docker and ensure the daemon is running.")
        return 2

    if mode in {"full", "lite"}:
        list_path = _default_list_path(mode)
    else:
        list_path = Path(mode).expanduser()

    try:
        images = _load_images_from_txt(list_path)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Failed to load image list: {e}")
        return 2

    if not images:
        print(f"⚠️ No images found in {list_path}")
        return 0

    print("=" * 60)
    print("ACE-Bench pre-pull images")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"List: {list_path}")
    print(f"Images: {len(images)}")
    print(f"Concurrency: {n_concurrent}")

    results: list[PullResult] = []
    if n_concurrent == 1:
        for idx, image in enumerate(images, 1):
            print("\n" + "-" * 60)
            print(f"[{idx}/{len(images)}] Pulling {image}")
            print("-" * 60)
            res = _docker_pull(image)
            results.append(res)
            if res.ok:
                print(f"✅ OK: {image}")
            else:
                print(f"❌ FAIL: {image} ({res.error})")
    else:
        # With concurrency > 1, capture docker output to avoid interleaved progress logs.
        results_by_image: dict[str, PullResult] = {}
        with ThreadPoolExecutor(max_workers=n_concurrent) as ex:
            futs = {
                ex.submit(_docker_pull, image, capture_output=True): image
                for image in images
            }
            done = 0
            total = len(images)
            for fut in as_completed(futs):
                image = futs[fut]
                done += 1
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    res = PullResult(image=image, ok=False, error=str(e))
                results_by_image[image] = res
                status = "OK" if res.ok else "FAIL"
                print(f"[{done}/{total}] {status}: {image}")

        # Rebuild results list in the same order as input.
        results = [results_by_image[i] for i in images]

    failed = [r for r in results if not r.ok]
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Success: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print(f"❌ Failed:  {len(failed)}")
        for r in failed:
            print(f"  - {r.image}: {r.error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
