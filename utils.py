import torch
import numpy as np
import numba

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


@numba.jit(nopython=True, nogil=True)
def gen_oracle_map(feat, ind, w, h):
  # feat: B x maxN x featDim
  # ind: B x maxN
  batch_size = feat.shape[0]
  max_objs = feat.shape[1]
  feat_dim = feat.shape[2]
  out = np.zeros((batch_size, feat_dim, h, w), dtype=np.float32)
  vis = np.zeros((batch_size, h, w), dtype=np.uint8)
  ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  for i in range(batch_size):
    queue_ind = np.zeros((h*w*2, 2), dtype=np.int32)
    queue_feat = np.zeros((h*w*2, feat_dim), dtype=np.float32)
    head, tail = 0, 0
    for j in range(max_objs):
      if ind[i][j] > 0:
        x, y = ind[i][j] % w, ind[i][j] // w
        out[i, :, y, x] = feat[i][j]
        vis[i, y, x] = 1
        queue_ind[tail] = x, y
        queue_feat[tail] = feat[i][j]
        tail += 1
    while tail - head > 0:
      x, y = queue_ind[head]
      f = queue_feat[head]
      head += 1
      for (dx, dy) in ds:
        xx, yy = x + dx, y + dy
        if xx >= 0 and yy >= 0 and xx < w and yy < h and vis[i, yy, xx] < 1:
          out[i, :, yy, xx] = f
          vis[i, yy, xx] = 1
          queue_ind[tail] = xx, yy
          queue_feat[tail] = f
          tail += 1
  return out
