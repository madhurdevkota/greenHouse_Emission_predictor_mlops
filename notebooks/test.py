## include all required libraries here
import sys
from pathlib import Path

ROOT = Path.cwd().parent  ## notebooks -> repo root

SRC  = ROOT / 'src'               # repo root/src
# SRC
sys.path.insert(  0,  str(SRC)  )
## import local utils
import api.utils as utils


import_block = """
import os
from copy import deepcopy
import glob
import pathlib
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
import reverse_geocoder as rg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
"""
result, failed_import = utils.import_lib(import_block)

DIRPATH = r'data/raw'
os.path.join( DIRPATH, '**/' )