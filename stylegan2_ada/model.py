import dnnlib
import numpy as np
import torch
import legacy

class StyleGAN(object):
    def __init__(self, 
                 model_path:str = None, 
                 device: str = 'cpu',
                 trunc: float = 1.0,) -> None:
        self.model_path = model_path
        self.device = device
        self.G = None
        self.W = None
        self.trunc = trunc
    
    # Modified from https://github.com/skimai/DragGAN/blob/main/draggan.py
    # 这里实现模型的加载以及添加hook(为了导出中间层特征)
    def load_ckpt(self, model_path) -> None:
        # 这种模型加载方式太不友好了
        with dnnlib.util.open_url(model_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad_(False)

        self.register_hook()

    def register_hook(self):
        # 新建变量用来储存中间层特征
        self.G.__setattr__("activations", None)

        # 新建hook，并apply到生成器的第7层，后面draggan需要用到
        def hook(module, input, output):
            self.G.activations = output
        for i, (name, module) in enumerate(self.G.synthesis.named_children()):
            if i == 6:
                print("Registering hook for:", name)
                module.register_forward_hook(hook)
    
    # 根据seed生成w向量
    def gen_w(self, seed):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(torch.float32).to(self.device)
        W = self.G.mapping(
            z,
            None,
            truncation_psi=self.trunc,
            truncation_cutoff=None,
        )
        return W
    
    # 根据w向量生成图像
    def gen_img(self, w):
        if not isinstance(w, torch.Tensor):
            w = torch.from_numpy(w).to(self.device)
        img = self.G.synthesis(w, noise_mode="const", force_fp32=True)
        return img, self.G.activations[0]


