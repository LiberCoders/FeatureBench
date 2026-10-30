from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.conda import CondaSourceAdapter
from codeworld.sources.pypi import PypiSourceAdapter
from codeworld.sources.pytorch import PytorchCpuSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(PypiSourceAdapter())
    registry.register(PytorchCpuSourceAdapter())
    registry.register(CondaSourceAdapter())
    registry.register(
        ArgumentSourceAdapter(
            name="python",
            services=frozenset({"uv_dist", "uv_python", "miniforge"}),
            arguments={
                "uv_dist": ("UV_INSTALLER_GITHUB_BASE_URL",),
                "uv_python": ("UV_PYTHON_INSTALL_MIRROR",),
                "miniforge": ("MINIFORGE_DIST_URL",),
            },
            hints={
                "uv_dist": (
                    "UV_INSTALLER_GITHUB_BASE_URL",
                    "astral-sh/uv/releases/download",
                    "uv-x86_64",
                    "uv-aarch64",
                ),
                "uv_python": ("uv python", "python-build-standalone"),
                "miniforge": ("miniforge",),
            },
            transforms={},
        )
    )
