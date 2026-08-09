
from .contractions import *
from .core import *
from .time_evolution import *
from .special_ttn import *
from .ttno.ttno_class import *
from .ttns.ttns import *
from .operators import *
from .dmrg import *

# A star import copies EVERY public name of a subpackage, including the names its own
# ``__init__`` bound to its submodules. Two of those collide with subpackage names here --
# ``time_evolution.time_evolution`` and ``special_ttn.util`` -- and, arriving later, won the
# attribute: ``import pytreenet.util`` handed back the 21-line ``special_ttn.util`` module
# instead of the ``pytreenet.util`` package, and ``import pytreenet.time_evolution`` handed
# back the ``time_evolution`` MODULE instead of the package. Re-bind both to the packages --
# via ``import_module``, which reads ``sys.modules`` directly; ``from . import x`` would just
# look the shadowed attribute up again and change nothing.
from importlib import import_module as _import_module

time_evolution = _import_module(__name__ + ".time_evolution")
util = _import_module(__name__ + ".util")
del _import_module
