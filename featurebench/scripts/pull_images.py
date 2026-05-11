#!/usr/bin/env python3
"""Pre-pull Docker images required by FeatureBench before running infer.

Usage:
  python featurebench/scripts/pull_images.py --mode full
  python featurebench/scripts/pull_images.py --mode lite
  python featurebench/scripts/pull_images.py --mode fast
  python featurebench/scripts/pull_images.py --mode /path/to/images.txt

Mode:
  - full: use featurebench/resources/constants/full_images.txt
  - lite: use featurebench/resources/constants/lite_images.txt
  - fast: use featurebench/resources/constants/fast_images.txt
  - custom path: any txt file where each line is an image reference

Notes:
  - Empty lines and lines starting with '#' are ignored.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from featurebench.utils.docker_images import (
    IMAGE_PREFIX_ENV_VAR,
    canonical_docker_hub_image_name,
    get_image_prefix,
    normalize_image_name,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_list_path(mode: str) -> Path:
    root = _repo_root()
    if mode == "full":
        return root / "featurebench" / "resources" / "constants" / "full_images.txt"
    if mode == "lite":
        return root / "featurebench" / "resources" / "constants" / "lite_images.txt"
    if mode == "fast":
        return root / "featurebench" / "resources" / "constants" / "fast_images.txt"
    raise ValueError(f"Unknown mode: {mode}")


@dataclass(frozen=True)
class ImagePullSpec:
    image: str
    official_tag: str | None = None


def _make_image_pull_spec(raw: str) -> ImagePullSpec | None:
    image = normalize_image_name(raw)
    if not image:
        return None

    official_tag = canonical_docker_hub_image_name(raw)
    if official_tag == image:
        official_tag = None

    return ImagePullSpec(image=image, official_tag=official_tag)


def _load_images_from_txt(path: Path) -> list[ImagePullSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Image list file not found: {path}")

    images: list[ImagePullSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        spec = _make_image_pull_spec(line)
        if spec is not None:
            images.append(spec)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[ImagePullSpec] = []
    for spec in images:
        if spec.image not in seen:
            uniq.append(spec)
            seen.add(spec.image)
    return uniq


@dataclass
class PullResult:
    image: str
    ok: bool
    official_tag: str | None = None
    tagged_official: bool = False
    error: str | None = None
    output: str | None = None


def _run_docker_command(args: list[str], *, capture_output: bool) -> tuple[int, str | None]:
    proc = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    return proc.returncode, proc.stdout if capture_output else None


def _docker_pull(spec: ImagePullSpec, *, capture_output: bool = False) -> PullResult:
    try:
        pull_code, pull_output = _run_docker_command(
            ["docker", "pull", spec.image],
            capture_output=capture_output,
        )
        if pull_code != 0:
            return PullResult(
                image=spec.image,
                ok=False,
                official_tag=spec.official_tag,
                error=f"docker pull exited with {pull_code}",
                output=pull_output,
            )

        if spec.official_tag:
            tag_code, tag_output = _run_docker_command(
                ["docker", "tag", spec.image, spec.official_tag],
                capture_output=capture_output,
            )
            output = "\n".join(part for part in (pull_output, tag_output) if part)
            if tag_code != 0:
                return PullResult(
                    image=spec.image,
                    ok=False,
                    official_tag=spec.official_tag,
                    error=f"docker tag exited with {tag_code}",
                    output=output or None,
                )
            return PullResult(
                image=spec.image,
                ok=True,
                official_tag=spec.official_tag,
                tagged_official=True,
                output=output or None,
            )

        return PullResult(
            image=spec.image,
            ok=True,
            official_tag=spec.official_tag,
            output=pull_output,
        )

    except FileNotFoundError:
        return PullResult(
            image=spec.image,
            ok=False,
            official_tag=spec.official_tag,
            error="docker command not found",
        )
    except Exception as e:  # noqa: BLE001
        return PullResult(
            image=spec.image,
            ok=False,
            official_tag=spec.official_tag,
            error=str(e),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-pull Docker images for FeatureBench")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        help="full | lite | fast | /path/to/images.txt (default: full)",
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

    if mode in {"full", "lite", "fast"}:
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
    print("FeatureBench pre-pull images")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"List: {list_path}")
    print(f"Images: {len(images)}")
    print(f"Concurrency: {n_concurrent}")
    print(f"Image prefix: {get_image_prefix()} (env: {IMAGE_PREFIX_ENV_VAR})")
    official_tag_count = sum(1 for spec in images if spec.official_tag)
    print(f"Official Docker Hub tags: {official_tag_count} to add locally")

    results: list[PullResult] = []
    if n_concurrent == 1:
        for idx, spec in enumerate(images, 1):
            print("\n" + "-" * 60)
            print(f"[{idx}/{len(images)}] Pulling {spec.image}")
            print("-" * 60)
            res = _docker_pull(spec)
            results.append(res)
            if res.ok:
                print(f"✅ OK: {res.image}")
                if res.tagged_official:
                    print(f"🏷️  TAG: {res.official_tag}")
            else:
                print(f"❌ FAIL: {res.image} ({res.error})")
    else:
        # With concurrency > 1, capture docker output to avoid interleaved progress logs.
        results_by_image: dict[str, PullResult] = {}
        with ThreadPoolExecutor(max_workers=n_concurrent) as ex:
            futs = {
                ex.submit(_docker_pull, spec, capture_output=True): spec.image
                for spec in images
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
                status = "OK+TAG" if res.tagged_official else ("OK" if res.ok else "FAIL")
                print(f"[{done}/{total}] {status}: {image}")

        # Rebuild results list in the same order as input.
        results = [results_by_image[spec.image] for spec in images]

    failed = [r for r in results if not r.ok]
    tagged = [r for r in results if r.tagged_official]
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Success: {len(results) - len(failed)}/{len(results)}")
    print(f"🏷️  Official tags added: {len(tagged)}")
    if failed:
        print(f"❌ Failed:  {len(failed)}")
        for r in failed:
            print(f"  - {r.image}: {r.error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
