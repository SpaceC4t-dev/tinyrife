import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

torch.set_num_threads(4)
torch.set_grad_enabled(False)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
        ),
        nn.LeakyReLU(0.2, True),
    )

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask

def warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)
    tenFlow = torch.cat([tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1)
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True)

class IFNet(nn.Module):
    def __init__(self, scale=1, ensemble=False):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](torch.cat((img0, img1, timestep), 1), None, scale=self.scale_list[i])
                if self.ensemble:
                    f1, m1 = block[i](torch.cat((img1, img0, 1 - timestep), 1), None, scale=self.scale_list[i])
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = block[i](
                    torch.cat((warped_img0, warped_img1, timestep, mask), 1), flow, scale=self.scale_list[i]
                )
                if self.ensemble:
                    f1, m1 = block[i](
                        torch.cat((warped_img1, warped_img0, 1 - timestep, -mask), 1),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=self.scale_list[i],
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2
                flow = flow + f0
                mask = mask + m0
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            warped_img1 = warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        return merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])

def run_bench(iters=10):
    model = IFNet()
    state_dict = torch.load('/tmp/flownet_v4.6.pkl', map_location='cpu')
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    im0 = np.array(Image.open('/home/user/repos/playground/frame0.png').convert('RGB')).astype(np.float32) / 255.0
    im1 = np.array(Image.open('/home/user/repos/playground/frame1.png').convert('RGB')).astype(np.float32) / 255.0

    t0 = torch.from_numpy(im0).permute(2, 0, 1).unsqueeze(0)
    t1 = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0)
    H, W = t0.shape[2], t0.shape[3]
    timestep = torch.full((1, 1, H, W), 0.5)

    pw, ph = W, H
    tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float)
    tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
    tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
    backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    # Warmup
    out = model(t0, t1, timestep, tenFlow_div, backwarp_tenGrid)

    times = []
    for _ in range(iters):
        t_start = time.perf_counter()
        out = model(t0, t1, timestep, tenFlow_div, backwarp_tenGrid)
        t_end = time.perf_counter()
        times.append((t_end - t_start) * 1000.0)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)

    print(f"Practical-RIFE (PyTorch CPU, 4 threads, {iters} runs): Mean = {mean_ms:.2f} ms, Std = {std_ms:.2f} ms, Min = {min_ms:.2f} ms, Max = {max_ms:.2f} ms")

    out_np = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out_np).save('/tmp/bench_pytorch_out.png')

    ref = np.array(Image.open('/home/user/repos/playground/ref_out.png').convert('RGB'))
    diff = np.abs(ref.astype(int) - out_np.astype(int))
    mse = np.mean((ref.astype(float) - out_np.astype(float))**2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"PyTorch Output Fidelity: Max diff = {diff.max()}, Mean abs diff = {diff.mean():.6f}, PSNR = {psnr:.2f} dB")

if __name__ == '__main__':
    run_bench(10)
