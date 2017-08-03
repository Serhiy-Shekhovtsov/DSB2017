import numpy as np
import torch
import pynvml


class SplitComb():
    def __init__(self, side_len=144, margin=32, **kwargs):
        self.side_len = side_len
        self.margin = margin
        self.max_stride = kwargs['max_stride']
        self.stride = kwargs['stride']
        self.pad_value = kwargs['pad_value']

    def split(self, data, side_len=None, max_stride=None, margin=None):
        if side_len is None:
            side_len = self.side_len

        if max_stride is None:
            max_stride = self.max_stride

        if margin is None:
            margin = self.margin

        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [
            [0, 0],
            [margin, nz * side_len - z + margin],
            [margin, nh * side_len - h + margin],
            [margin, nw * side_len - w + margin]]

        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzhw

    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):
        if side_len is None:
            side_len = self.side_len

        if stride is None:
            stride = self.stride

        if margin is None:
            margin = self.margin

        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw

        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len //= stride
        margin //= stride

        splits = []

        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones(
            (
                nz * side_len, nh * side_len, nw * side_len,
                splits[0].shape[3], splits[0].shape[4]
            ), np.float32)

        idx = 0

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][
                        margin:margin + side_len, margin:margin + side_len,
                        margin:margin + side_len]

                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output


def getFreeId():
    pynvml.nvmlInit()

    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5 * (float(use.gpu + float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []

    for i in range(deviceCount):
        if getFreeRatio(i) < 70:
            available.append(i)

    gpus = ''

    for g in available:
        gpus = gpus + str(g) + ', '

    return gpus[:-1]


def split4(data, max_stride, margin):
    splits = []
    data = torch.Tensor.numpy(data)
    _, c, z, h, w = data.shape

    w_width = np.ceil(float(w // 2 + margin) / max_stride).astype('int') * max_stride
    h_width = np.ceil(float(h // 2 + margin) / max_stride).astype('int') * max_stride
    pad = int(np.ceil(float(z) / max_stride) * max_stride) - z
    leftpad = pad // 2
    pad = [[0, 0], [0, 0], [leftpad, pad - leftpad], [0, 0], [0, 0]]
    data = np.pad(data, pad, 'constant', constant_values=-1)
    data = torch.from_numpy(data)
    splits.append(data[:, :, :, :h_width, :w_width])
    splits.append(data[:, :, :, :h_width, -w_width:])
    splits.append(data[:, :, :, -h_width:, :w_width])
    splits.append(data[:, :, :, -h_width:, -w_width:])
    return torch.cat(splits, 0)


def combine4(output, h, w):
    splits = []

    for i in range(len(output)):
        splits.append(output[i])

    output = np.zeros(
        (splits[0].shape[0], h, w, splits[0].shape[3], splits[0].shape[4]),
        np.float32)

    h0 = output.shape[1] // 2
    h1 = output.shape[1] - h0
    w0 = output.shape[2] // 2
    w1 = output.shape[2] - w0

    splits[0] = splits[0][:, :h0, :w0, :, :]
    output[:, :h0, :w0, :, :] = splits[0]

    splits[1] = splits[1][:, :h0, -w1:, :, :]
    output[:, :h0, -w1:, :, :] = splits[1]

    splits[2] = splits[2][:, -h1:, :w0, :, :]
    output[:, -h1:, :w0, :, :] = splits[2]

    splits[3] = splits[3][:, -h1:, -w1:, :, :]
    output[:, -h1:, -w1:, :, :] = splits[3]
    return output


def split8(data, max_stride, margin):
    splits = []

    if isinstance(data, np.ndarray):
        c, z, h, w = data.shape
    else:
        _, c, z, h, w = data.size()

    z_width = np.ceil(float(z // 2 + margin) / max_stride).astype('int') * max_stride
    w_width = np.ceil(float(w // 2 + margin) / max_stride).astype('int') * max_stride
    h_width = np.ceil(float(h // 2 + margin) / max_stride).astype('int') * max_stride

    for zz in [[0, z_width], [-z_width, None]]:
        for hh in [[0, h_width], [-h_width, None]]:
            for ww in [[0, w_width], [-w_width, None]]:
                if isinstance(data, np.ndarray):
                    splits.append(data[np.newaxis, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
                else:
                    splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

    if isinstance(data, np.ndarray):
        return np.concatenate(splits, 0)
    else:
        return torch.cat(splits, 0)


def combine8(output, z, h, w):
    splits = []

    for i in range(len(output)):
        splits.append(output[i])

    output = np.zeros(
        (z, h, w, splits[0].shape[3], splits[0].shape[4]), np.float32)

    z_width = z // 2
    h_width = h // 2
    w_width = w // 2
    i = 0

    for zz in [[0, z_width], [z_width - z, None]]:
        for hh in [[0, h_width], [h_width - h, None]]:
            for ww in [[0, w_width], [w_width - w, None]]:
                res = splits[i][zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = res
                i = i + 1

    return output


def split16(data, max_stride, margin):
    splits = []
    _, c, z, h, w = data.size()

    z_width = np.ceil(float(z // 4 + margin) / max_stride).astype('int') * max_stride
    z_pos = [z * 3 // 8 - z_width // 2, z * 5 // 8 - z_width // 2]
    h_width = np.ceil(float(h // 2 + margin) / max_stride).astype('int') * max_stride
    w_width = np.ceil(float(w // 2 + margin) / max_stride).astype('int') * max_stride

    for zz in [
        [0, z_width], [z_pos[0], z_pos[0] + z_width],
        [z_pos[1], z_pos[1] + z_width], [-z_width, None]
    ]:
        for hh in [[0, h_width], [-h_width, None]]:
            for ww in [[0, w_width], [-w_width, None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

    return torch.cat(splits, 0)


def combine16(output, z, h, w):
    splits = []

    for i in range(len(output)):
        splits.append(output[i])

    output = np.zeros(
        (z, h, w, splits[0].shape[3], splits[0].shape[4]), np.float32)

    z_width = z // 4
    h_width = h // 2
    w_width = w // 2
    splitzstart = splits[0].shape[0] // 2 - z_width // 2
    i = 0

    for zz, zz2 in zip(
        [
            [0, z_width], [z_width, z_width * 2], [z_width * 2, z_width * 3],
            [z_width * 3 - z, None]
        ],
        [
            [0, z_width], [splitzstart, z_width + splitzstart],
            [splitzstart, z_width + splitzstart], [z_width * 3 - z, None]
        ]
    ):
        for hh in [[0, h_width], [h_width - h, None]]:
            for ww in [[0, w_width], [w_width - w, None]]:
                res = splits[i][zz2[0]:zz2[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = res
                i = i + 1

    return output


def split32(data, max_stride, margin):
    splits = []
    _, c, z, h, w = data.size()

    z_width = np.ceil(float(z // 2 + margin) / max_stride).astype('int') * max_stride
    w_width = np.ceil(float(w // 4 + margin) / max_stride).astype('int') * max_stride
    h_width = np.ceil(float(h // 4 + margin) / max_stride).astype('int') * max_stride

    w_pos = [w * 3 // 8 - w_width // 2, w * 5 // 8 - w_width // 2]
    h_pos = [h * 3 // 8 - h_width // 2, h * 5 // 8 - h_width // 2]

    for zz in [[0, z_width], [-z_width, None]]:
        for hh in [
            [0, h_width], [h_pos[0], h_pos[0] + h_width],
            [h_pos[1], h_pos[1] + h_width], [-h_width, None]
        ]:
            for ww in [
                [0, w_width], [w_pos[0], w_pos[0] + w_width],
                [w_pos[1], w_pos[1] + w_width], [-w_width, None]
            ]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

    return torch.cat(splits, 0)


def combine32(splits, z, h, w):
    output = np.zeros(
        (z, h, w, splits[0].shape[3], splits[0].shape[4]), np.float32)

    z_width = int(np.ceil(float(z) / 2))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splithstart = splits[0].shape[1] // 2 - h_width // 2
    splitwstart = splits[0].shape[2] // 2 - w_width // 2
    i = 0

    for zz in [[0, z_width], [z_width - z, None]]:

        for hh, hh2 in zip(
            [
                [0, h_width], [h_width, h_width * 2], [h_width * 2, h_width * 3],
                [h_width * 3 - h, None]
            ],
            [
                [0, h_width], [splithstart, h_width + splithstart],
                [splithstart, h_width + splithstart], [h_width * 3 - h, None]
            ]
        ):

            for ww, ww2 in zip(
                [
                    [0, w_width], [w_width, w_width * 2],
                    [w_width * 2, w_width * 3], [w_width * 3 - w, None]
                ],
                [
                    [0, w_width], [splitwstart, w_width + splitwstart],
                    [splitwstart, w_width + splitwstart], [w_width * 3 - w, None]
                ]
            ):
                res = splits[i][zz[0]:zz[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = res
                i = i + 1

    return output


def split64(data, max_stride, margin):
    splits = []
    _, c, z, h, w = data.size()

    z_width = np.ceil(float(z // 4 + margin) / max_stride).astype('int') * max_stride
    w_width = np.ceil(float(w // 4 + margin) / max_stride).astype('int') * max_stride
    h_width = np.ceil(float(h // 4 + margin) / max_stride).astype('int') * max_stride

    z_pos = [z * 3 // 8 - z_width // 2, z * 5 // 8 - z_width // 2]
    w_pos = [w * 3 // 8 - w_width // 2, w * 5 // 8 - w_width // 2]
    h_pos = [h * 3 // 8 - h_width // 2, h * 5 // 8 - h_width // 2]

    for zz in [
        [0, z_width], [z_pos[0], z_pos[0] + z_width], [z_pos[1], z_pos[1] + z_width],
        [-z_width, None]
    ]:
        for hh in [
            [0, h_width], [h_pos[0], h_pos[0] + h_width],
            [h_pos[1], h_pos[1] + h_width], [-h_width, None]
        ]:
            for ww in [
                [0, w_width], [w_pos[0], w_pos[0] + w_width],
                [w_pos[1], w_pos[1] + w_width], [-w_width, None]
            ]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

    return torch.cat(splits, 0)


def combine64(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])

    output = np.zeros(
        (z, h, w, splits[0].shape[3], splits[0].shape[4]), np.float32)

    z_width = int(np.ceil(float(z) / 4))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splitzstart = splits[0].shape[0] // 2 - z_width // 2
    splithstart = splits[0].shape[1] // 2 - h_width // 2
    splitwstart = splits[0].shape[2] // 2 - w_width // 2

    i = 0

    for zz, zz2 in zip(
        [
            [0, z_width], [z_width, z_width * 2], [z_width * 2, z_width * 3],
            [z_width * 3 - z, None]
        ],
        [
            [0, z_width], [splitzstart, z_width + splitzstart],
            [splitzstart, z_width + splitzstart], [z_width * 3 - z, None]
        ]
    ):
        for hh, hh2 in zip(
            [
                [0, h_width], [h_width, h_width * 2], [h_width * 2, h_width * 3],
                [h_width * 3 - h, None]
            ],
            [
                [0, h_width], [splithstart, h_width + splithstart],
                [splithstart, h_width + splithstart], [h_width * 3 - h, None]
            ]
        ):
            for ww, ww2 in zip(
                [
                    [0, w_width], [w_width, w_width * 2], [w_width * 2, w_width * 3],
                    [w_width * 3 - w, None]
                ],
                [
                    [0, w_width], [splitwstart, w_width + splitwstart],
                    [splitwstart, w_width + splitwstart], [w_width * 3 - w, None]
                ]
            ):
                res = splits[i][zz2[0]:zz2[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = res
                i = i + 1

    return output
