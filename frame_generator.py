import math
import sys
import numpy as np
from tqdm import tqdm

from tbe import TemporalBinaryEncoding
from src.io.psee_loader import PSEELoader
from tbe import TemporalBinaryEncoding


def encode_video_tbe(N: int, 
                    video: PSEELoader, 
                    delta: int = 100000) -> np.array:
    
    print("Starting Temporal Binary Encoding...")

    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    height, width = video.get_size()
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float32, (height, width))])
    
    samplePerVideo = math.ceil((video.total_time() / delta) / N)
    accumulation_mat = np.zeros((N, height, width))
    tbe_array = np.zeros(samplePerVideo, dtype=data_type)
    encoder = TemporalBinaryEncoding(N, width, height)

    i = 0
    j = 0
    startTimestamp = 0  # milliseconds
    endTimestamp = 0    # milliseconds

    pbar = tqdm(total = samplePerVideo, file = sys.stdout)
    while not video.done:
        i = (i + 1) % N
        # Load next 1ms events from the video
        events = video.load_delta_t(delta)
        f = np.zeros(video.get_size())
        for e in events:
            # Evaluate presence/absence of event for
            # a certain pixel
            f[e['y'], e['x']] = 1

        accumulation_mat[i, ...] = f

        if i == N - 1:
            endTimestamp += (N * delta) / 1000
            tbe = encoder.encode(accumulation_mat)
            tbe_array[j]['startTs'] = startTimestamp
            tbe_array[j]['endTs'] = endTimestamp
            tbe_array[j]['frame'] = tbe
            j += 1
            startTimestamp += (N * delta) / 1000
            pbar.update(1)
    
    pbar.close()
    return tbe_array


def get_frame_BB(frame: np.array, BB_array: np.array) -> np.array:
    """
    @brief: Associates to an encoded video frame
            a list of bounding boxes with timestamp included in 
            start/end timestamp of the frame. 
    @param: frame - Encoded frame with the following structure:
                    [{'startTs': startTs}, {'endTs': endTs}, {'frame': frame}]
                    (i.e. as the one returned from the encoders fuctions)
    @param: BB_array - Bounding Boxes array, 
                       loaded from the GEN1 .npy arrays
    @return: The associated BBoxes.
    """

    associated_bb = []
    for bb in BB_array:
        # Convert timestamp to milliseconds
        timestamp = bb[0] / 1000
        startTime = frame['startTs']
        endTime = frame['endTs']
        if timestamp >= startTime and timestamp <= endTime:
            associated_bb.append(bb[5])
            break
        # Avoid useless iterations
        if timestamp > endTime:
            break
    
    return np.array(associated_bb)


def get_frame(file_path):

    video = PSEELoader(file_path)
    frame = encode_video_tbe(4, video)
    return frame

def get_class(file_path, frames):
    bbox = np.load(file_path)
    associated_bb = []
    for frame in frames:
        array = get_frame_BB(frame, bbox)
        associated_bb.append(array)
    
    return associated_bb
