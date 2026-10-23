"""Deprecated import alias for historical regression tests only."""

import sys
from ship_perception.legacy.ros1 import ros_input_adapter as _legacy

sys.modules[__name__] = _legacy
