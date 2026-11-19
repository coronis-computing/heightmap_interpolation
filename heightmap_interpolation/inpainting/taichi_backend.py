# Copyright (c) 2026 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)
"""Single entry point to the Taichi-compatible GPU backend.

Upstream Taichi (``taichi``) is no longer maintained and does not support
Python >= 3.14. Taichi Forge (``pip install taichi-forge``) is a community fork
that keeps the Taichi API and ships wheels for Python 3.10-3.14. Depending on the
release, Taichi Forge is importable either as ``taichi_forge`` or as ``taichi``,
so we try both. Only one of them should be installed in a given environment.

Usage::

    from heightmap_interpolation.inpainting.taichi_backend import ti, TAICHI_AVAILABLE
"""

ti = None
TAICHI_AVAILABLE = False
TAICHI_IMPORT_ERROR = None

try:
    import taichi_forge as ti  # noqa: F811

    TAICHI_AVAILABLE = True
except ImportError:
    try:
        import taichi as ti  # noqa: F811

        TAICHI_AVAILABLE = True
    except Exception as e:  # ImportError, or a broken binary on this platform
        TAICHI_IMPORT_ERROR = e


def require_taichi():
    """Returns the taichi module, or raises an informative ImportError."""
    if not TAICHI_AVAILABLE:
        raise ImportError(
            "The GPU backend requires a Taichi-compatible package, which could not be "
            "imported (" + str(TAICHI_IMPORT_ERROR) + "). Install it with: "
            "pip install heightmap_interpolation[gpu]"
        ) from TAICHI_IMPORT_ERROR
    return ti
