"""
Specifying the parameters for the matplotlib figures produced by visualize_gradients.py
"""

UR_CORNER_PARAMS = {'axis':'both', 'which':'both', 'left':'off', 'right':'off', 'bottom':'off', 'top': 'off', 'labelbottom':'off','labelleft':'off', 'labeltop':'off'}

BOTTOM_EDGE_PARAMS = UR_CORNER_PARAMS.copy()
BOTTOM_EDGE_PARAMS['bottom'] = 'on'
BOTTOM_EDGE_PARAMS['labelbottom'] = 'on'

LEFT_EDGE_PARAMS = UR_CORNER_PARAMS.copy()
BOTTOM_EDGE_PARAMS['left'] = 'on'
BOTTOM_EDGE_PARAMS['labelleft'] = 'on'

LL_CORNER_PARAMS = UR_CORNER_PARAMS.copy()
LL_CORNER_PARAMS['left'] = 'on'
LL_CORNER_PARAMS['bottom'] = 'on'
LL_CORNER_PARAMS['labelleft'] = 'on'
LL_CORNER_PARAMS['labelbottom'] = 'on'

FIG_SIZE = (12,12)
CMAP = "RdBu"
V_MIN = -1
V_MAX = 1
DPI = 600