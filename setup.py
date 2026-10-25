"""Everything is in pyproject.toml; this only makes the wheel per-interpreter.

hk._mlir is the MLIR Python bindings built for one CPython (nanobind
extension modules), so a wheel carrying it is cp3XX-cp3XX-<platform>, not
py3-none-any. setuptools tags a wheel that way only for a distribution with
extension modules, and the bindings are built by CMake, not by setuptools --
hence the declaration. Build with scripts/build_wheel.sh, which stages the
headers and checks that hk._mlir matches the interpreter.

The one other thing done here is the package index's description. README.md's
links are relative, which resolve on GitHub and nowhere on PyPI, so the
description is README.md with every relative link pointed at the same file in
the release's tag (pypi_readme below; tests/hk/ir/test_pypi_readme.py checks it).
"""

import pathlib
import re

from setuptools import setup
from setuptools.dist import Distribution

REPO = "https://github.com/vivienfanghuagood/hk-dsl"
RAW = "https://raw.githubusercontent.com/vivienfanghuagood/hk-dsl"

# images first, ![alt](target "title"), then every remaining ](target "title");
# the second also catches a badge's outer link, [![alt](img)](target)
_IMAGE = re.compile(r'(!)\[[^\]]*\]\(([^)\s]+)((?:\s+"[^"]*")?)\)')
_LINK = re.compile(r'()\]\(([^)\s]+)((?:\s+"[^"]*")?)\)')


def pypi_readme(text: str, version: str) -> str:
    """README.md with its relative links made absolute at tag v<version>.

    Pages go to github.com/.../blob/v<version>/<path> (GitHub redirects a
    directory to its tree view), images to raw.githubusercontent.com, and a
    bare #anchor to the README on GitHub, since the index's renderer gives
    headings no ids. Absolute URLs and mailto: are left alone.
    """
    tag = f"v{version}"

    def absolute(m):
        bang, target, title = m.groups()
        if re.match(r"[a-z][a-z0-9+.-]*:", target, re.I):
            return m.group(0)
        if target.startswith("#"):
            target = "README.md" + target
        path = target.lstrip("./")
        base = f"{RAW}/{tag}" if bang else f"{REPO}/blob/{tag}"
        head = m.group(0)[: m.group(0).index("](")]
        return f"{head}]({base}/{path}{title})"

    return _LINK.sub(absolute, _IMAGE.sub(absolute, text))


def _version() -> str:
    text = (pathlib.Path(__file__).parent / "pyproject.toml").read_text()
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    readme = (pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    setup(
        distclass=BinaryDistribution,
        long_description=pypi_readme(readme, _version()),
        long_description_content_type="text/markdown",
    )
