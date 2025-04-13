'''
# dataset process function
from PIL import Image, ImageOps
import numpy as np

def rgb_to_rmi(image):
    r, g, b = image.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    
    gb_max = np.maximum(g, b)
    gray_c = np.array(ImageOps.grayscale(image))
    combined = np.stack((r, gb_max, gray_c), axis=-1)
    
    return Image.fromarray(combined)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from other_module import FeatureExtractor  
# such as Adabins FeatureExtractor


class EIGM(nn.Module):
    def __init__(self, mu0=1.0, mu1=1.0, mu2=1.0):
        super(EIGM, self).__init__()
        self.mu0 = nn.Parameter(torch.tensor(mu0, dtype=torch.float32), requires_grad=True)
        self.mu1 = nn.Parameter(torch.tensor(mu1, dtype=torch.float32), requires_grad=True)
        self.mu2 = nn.Parameter(torch.tensor(mu2, dtype=torch.float32), requires_grad=True)
        self.feature_extractor = FeatureExtractor()

    def ULAP(self, image):
        r, g, b = image.split(1, dim=0)
        m = torch.maximum(g, b)
        d = self.mu0 + self.mu1 * m + self.mu2 * r
        return d.squeeze(0)

    def estimate_background_and_direct_signal(self, image, depth, depth_segments=10, bottom_percent=1):
        H, W = depth.shape
        depth_bins = torch.linspace(depth.min(), depth.max(), depth_segments + 1, device=depth.device)
        B_c_list = []
        D_c_list = []

        for i in range(depth_segments):
            start_depth = depth_bins[i]
            end_depth = depth_bins[i + 1]
            segment_mask = (depth >= start_depth) & (depth < end_depth)
            segment_image = image[:, segment_mask]
            segment_depth = depth[segment_mask]

            if segment_image.numel() == 0:
                continue

            min_rgb_values = segment_image.min(dim=0).values
            num_dark_pixels = int(len(min_rgb_values) * bottom_percent / 100)
            dark_pixels = min_rgb_values.topk(num_dark_pixels, largest=False).values

            B_c = dark_pixels.mean()
            B_c_list.append(B_c)

            segment_mean_rgb = segment_image.mean(dim=1)
            D_c = segment_mean_rgb - B_c
            D_c_list.append(D_c)

        return B_c_list, D_c_list

    def estimate_direct_signal_attenuation(self, D_c, depth_map, p=0.5, f=2.0):
        a_c = self.compute_local_average_color(D_c, depth_map, p)
        E_c = f * a_c
        beta_c_D = -torch.log(E_c) / depth_map
        return beta_c_D, E_c

    def compute_local_average_color(self, D_c, depth_map, p=0.5):
        B, C, H, W = D_c.shape
        a_c = torch.zeros_like(D_c)

        for y in range(H):
            for x in range(W):
                z = depth_map[0, 0, y, x]
                N_e = self.get_neighborhood(x, y, z, depth_map)
                a_prime_c = D_c[:, :, N_e[:, 0], N_e[:, 1]].mean(dim=2)
                a_c[:, :, y, x] = p * D_c[:, :, y, x] + (1 - p) * a_prime_c

        return a_c

    def get_neighborhood(self, x, y, z, depth_map, neighborhood_size=3):
        H, W = depth_map.shape[2], depth_map.shape[3]
        neighborhood_range = neighborhood_size // 2
        N_e = []

        for dy in range(-neighborhood_range, neighborhood_range + 1):
            for dx in range(-neighborhood_range, neighborhood_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and abs(depth_map[0, 0, ny, nx] - z) < 1:
                    N_e.append([ny, nx])

        return torch.tensor(N_e, dtype=torch.long, device=depth_map.device)

    def forward(self, image, depth):
        d = self.ULAP(image)
        B_c_list, D_c_list = self.estimate_background_and_direct_signal(image, depth)
        D_c_tensor = torch.stack(D_c_list).unsqueeze(0) 
        beta_c_D, E_c = self.estimate_direct_signal_attenuation(D_c_tensor, depth.unsqueeze(0).unsqueeze(0))
        J_c = E_c * d.unsqueeze(0).unsqueeze(0)
        features = self.feature_extractor(J_c)
        return features
