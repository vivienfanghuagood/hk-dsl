"""`python3 -m hk.autotune`.

A separate file rather than a `__main__` guard in `__init__.py`, because
`hk.ops` imports `hk.autotune` at import time to register its tuners: by the
time runpy went to execute `hk/autotune.py` as `__main__` the module was
already in `sys.modules` under its own name, so there were two module objects,
two REGISTRYs, and the one the command line could see was always empty. runpy
warns about exactly this and the warning is right.
"""

from . import main

raise SystemExit(main())
