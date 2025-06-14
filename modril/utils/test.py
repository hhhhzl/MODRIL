# import torch
# import os

# dir = os.path.dirname(os.path.realpath(__file__))

# traj = torch.load(os.path.join(dir, "..", "expert_datasets/pick_2000.pt"))
# print(traj["obs"].shape)  # ! 20311

import os
import os.path as osp
import numpy as np
import cv2

from rlf.exp_mgr.viz_utils import save_mp4

# --- stand-in for pytest’s monkeypatch fixture ---
class DummyVideoWriter:
    def __init__(self, path, fourcc, fps, dims):
        self.path = path
        self.frames = []
    def write(self, frame):
        self.frames.append(frame.copy())
    def release(self):
        open(self.path, "wb").close()

# patch OpenCV
cv2.VideoWriter = DummyVideoWriter
cv2.VideoWriter_fourcc = lambda *args: 0x1234

# --- test 1: simple frames ---
def test_simple():
    tmp = "tmp_vid"
    os.makedirs(tmp, exist_ok=True)
    # 5 random H×W×3 frames
    frames = [np.random.randint(0,255,(4,6,3),dtype=np.uint8) for _ in range(5)]
    save_mp4(frames, tmp, "foo", fps=24.0, should_print=True)
    out = osp.join(tmp, "foo.mp4")
    assert osp.exists(out), "output file missing"
    print("test_simple passed")

# --- test 2: batched frames ---
def test_batched():
    tmp = "tmp_vid2"
    # two batches of 2 frames each (shape (2,H,W,3))
    batch1 = np.zeros((2,3,5,3),dtype=np.uint8)
    batch2 = np.ones ((2,3,5,3),dtype=np.uint8)*255
    save_mp4([batch1, batch2], tmp, "bar", fps=30.0, should_print=False)
    out = osp.join(tmp, "bar.mp4")
    assert osp.exists(out), "output file missing"
    print("test_batched passed")

if __name__ == "__main__":
    test_simple()
    test_batched()
    print("All tests passed!")