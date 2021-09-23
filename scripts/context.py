import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(str(Path(__file__).parents[1])))
from visualdet3d.data import kitti
from visualdet3d.data import pipeline
from visualdet3d.networks import detectors
from visualdet3d.networks import pipelines