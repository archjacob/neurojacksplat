import torch
import torch.nn as nn
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class NeuroJackSplatRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg_color = torch.zeros(3, device="cuda")

    def forward(self, means, colors, opacities, scales, rots, view_matrix, proj_matrix, campos):
        settings = GaussianRasterizationSettings(
            image_height=224, image_width=224, tanfovx=0.5, tanfovy=0.5,
            bg=self.bg_color, scale_modifier=1.0,
            viewmatrix=view_matrix, projmatrix=proj_matrix,
            sh_degree=0, campos=campos, prefiltered=False, debug=False
        )
        # Ky rasterizer është tani i izoluar dhe "i blinduar"
        rasterizer = GaussianRasterizer(settings)
        render, _ = rasterizer(
            means3D=means,
            means2D=torch.zeros_like(means, requires_grad=True),
            opacities=torch.sigmoid(opacities),
            shs=torch.zeros((means.shape[0], 0, 3), device="cuda"),
            colors_precomp=colors,
            scales=torch.exp(scales),
            rotations=torch.nn.functional.normalize(rots, dim=-1)
        )
        return render