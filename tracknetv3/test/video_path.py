import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = Path(FILE.parents[1]) 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  

input_video_path = f"{str(ROOT)}/video/nycu/hscc_fps50_long1.mp4"
output_video_path = f"{input_video_path[:-4]}_masked.mp4"