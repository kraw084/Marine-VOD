import os
import sys

#Add main repo directory so we can inport YoloV5, YoloX, and TrackEval
proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if proj_dir not in sys.path:
    sys.path.append(proj_dir)