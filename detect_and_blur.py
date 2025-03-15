import cv2
from video_streamer import VideoStreamer

video_source = "data\\people_entering.mp4"

with VideoStreamer(video_path=video_source) as vs:
    vs.start_streaming()
