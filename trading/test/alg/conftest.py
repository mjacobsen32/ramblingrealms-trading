import sys
from unittest.mock import MagicMock

# Patch matplotlib before anything else
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = MagicMock()

# Patch plotly (used by vectorbt for interactive plots)
try:
    import plotly.io as pio

    pio.show = MagicMock()
except ImportError:
    pass

# Patch vectorbt's Figure.show if present
try:
    import vectorbt as vbt

    if hasattr(vbt, "Figure"):
        vbt.Figure.show = MagicMock()
except ImportError:
    pass
