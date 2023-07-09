import torch.nn.functional as torch_F
import cv2
from stylegan2_ada.model import StyleGAN
import utils
import copy
import torch

class UI_Backend(object):
    def __init__(self, device):
        self.device = device
        self.model_path = None
        # 这里定义StyleGAN
        self.generator = StyleGAN(device=device)
    
    # 加载模型参数
    def load_ckpt(self, model_path):
        self.generator.load_ckpt(model_path)
        self.model_path = model_path
        
    # 根据seed生成对应的图像
    def gen_img(self, seed):
        if self.model_path is not None:
            # 将opt设置为None, 表示开始一次新的优化
            self.optimizer = None
            
            # seed -> w -> image
            self.W = self.generator.gen_w(seed)
            img, self.init_F = self.generator.gen_img(self.W)
            
            # 将生成的图片转换成可ui支持的raw data
            t_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            t_img = cv2.resize(t_img, (512, 512))
            raw_img = (t_img / 2 + 0.5).clip(0, 1).reshape(-1)
            
            return raw_img
        else:
            # 如果模型还没加载则返回None
            return None
    
    def prepare_to_drag(self, init_pts, lr=2e-3):
        # 1. 备份初始图像的特征图 -> motion supervision和point tracking都需要用到
        self.F0_resized = torch_F.interpolate(self.init_F, 
                                              size=(512, 512), 
                                              mode="bilinear",
                                              align_corners=True).detach().clone()
        # 2. 备份初始点坐标 -> point tracking
        temp_init_pts_0 = copy.deepcopy(init_pts)
        self.init_pts_0 = torch.from_numpy(temp_init_pts_0).float().to(self.device)
        
        # 3. 将w向量的部分特征设置为可训练
        temp_W = self.W.cpu().numpy().copy()
        self.W = torch.from_numpy(temp_W).to(self.device).float()
        self.W.requires_grad_(False)
        
        self.W_layers_to_optimize = self.W[:, :6, :].detach().clone().requires_grad_(True)
        self.W_layers_to_fixed = self.W[:, 6:, :].detach().clone().requires_grad_(False)
        
        # 4. 初始化优化器
        self.optimizer = torch.optim.Adam([self.W_layers_to_optimize], lr=lr)
    
    # 进行一次迭代优化
    def drag(self, 
             _init_pts, _tar_pts, 
             atol=2,  # 允许的误差，2个像素
             r1=3, r2=13):
        init_pts = torch.from_numpy(_init_pts).float().to(self.device)
        tar_pts = torch.from_numpy(_tar_pts).float().to(self.device)
        
        # 如果起始点和目标点之间的像素误差足够小，则停止
        if torch.allclose(init_pts, tar_pts, atol=atol):
            return False, (None, None)
        
        self.optimizer.zero_grad()

        # 将latent的0:6设置成可训练,6:设置成不可训练 See Sec3.2
        W_combined = torch.cat([self.W_layers_to_optimize, self.W_layers_to_fixed], dim=1)
        # 前向推理
        new_img, _F = self.generator.gen_img(W_combined)
        # See, Sec 3.2 in paper, 计算motion supervision loss
        F_resized = torch_F.interpolate(_F, size=(512, 512), mode="bilinear", align_corners=True)
        loss = self.motion_supervision(
            F_resized,
            init_pts, tar_pts,
            r1=r1)
        
        loss.backward()
        self.optimizer.step()
        
        # 更新初始点 see Sec3.3 Point Tracking
        with torch.no_grad():
            # 以上过程会优化一次latent, 直接用新的latent生成图像，用于中间过程的显示
            new_img, F_for_point_tracking = self.generator.gen_img(W_combined)
            new_img = new_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            new_img = cv2.resize(new_img, (512, 512))
            new_raw_img = (new_img / 2 + 0.5).clip(0, 1).reshape(-1)
            
            F_for_point_tracking_resized = torch_F.interpolate(F_for_point_tracking, size=(512, 512), 
                                                               mode="bilinear", align_corners=True).detach()
            new_init_pts = self.point_tracking(F_for_point_tracking_resized, init_pts, r2=r2)
            
        print(f"Loss: {loss.item():0.4f}, tar pts: {tar_pts.cpu().numpy()}, new init pts: {new_init_pts.cpu().numpy()}")
        print('\n')
        
        return True, (new_init_pts.detach().clone().cpu().numpy(), tar_pts.detach().clone().cpu().numpy(), new_raw_img)
    
    # 计算motion supervision loss, 用来更新w，使图像中目标点邻域的特征与起始点领域的特征靠近
    def motion_supervision(self, 
                           F,
                           init_pts, 
                           tar_pts, 
                           r1=3):
        
        n = init_pts.shape[0]
        loss = 0.0
        for i in range(n):
            # 计算方向向量
            dir_vec = tar_pts[i] - init_pts[i]
            d_i = dir_vec / (torch.norm(dir_vec) + 1e-7)
            if torch.norm(d_i) > torch.norm(dir_vec):
                d_i = dir_vec
            # 创建一个圆形mask区域，以起始点为中心，r为半径，也就是原文的red circle(Fig.3)
            mask = utils.create_circular_mask(
                F.shape[2], F.shape[3], center=init_pts[i].tolist(), radius=r1
            ).to(self.device)
            # 将mask的index找到
            coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]
            # 根据之前计算的运动向量，移动mask的index, 也就是得到目标点的mask对应的index
            shifted_coordinates = coordinates + d_i[None]
            # 从特征图中，拿出mask区域的特征
            F_qi = F[:, :, mask] # [1, C, num_points]

            # 下面这一坨代码都是为了得到mask平移后新的位置的特征, 并且需要这个过程可微
            # 1. 将coordinates坐标系的原点平移到图像中心，为了grid_sample函数准备
            h, w = F.shape[2], F.shape[3]
            norm_shifted_coordinates = shifted_coordinates.clone()
            norm_shifted_coordinates[:, 0] = (2.0 * shifted_coordinates[:, 0] / (h - 1)) - 1
            norm_shifted_coordinates[:, 1] = (2.0 * shifted_coordinates[:, 1] / (w - 1)) - 1
            # 将norm_shifted_coordinates尺寸从[num_points, 2] -> [1, 1, num_points, 2] 
            norm_shifted_coordinates = norm_shifted_coordinates.unsqueeze(0).unsqueeze(0)
            norm_shifted_coordinates = norm_shifted_coordinates.clamp(-1, 1)
            # 2. 将[x, y] -> [y, x]，为了grid_sample函数准备
            norm_shifted_coordinates = norm_shifted_coordinates.flip(-1)
            # 3. 执行grid_sample，拿到平移后mask对应的特征
            F_qi_plus_di = torch_F.grid_sample(F, norm_shifted_coordinates, mode="bilinear", align_corners=True)
            F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

            # 监督移动前后的特征要一致
            loss += torch_F.l1_loss(F_qi.detach(), F_qi_plus_di)
            # 同时监督特征图上, mask外的特征要一致 暂不支持
        
        return loss

    # 目的是更新初始点,因为图像通过motion_supervision已经发生了变化
    # init_pts -> new init_pts -> ... -> tar_pts
    def point_tracking(self, 
                       F,
                       init_pts, 
                       r2):
        n = init_pts.shape[0]
        new_init_pts = torch.zeros_like(init_pts)
        '''
        为什么要有这一步: 
            motion_supervision更新了图像, 之前的初始点不知道具体移动到哪里了
        如何找到最新的初始点:  
            在老的初始点附近划定一个区域, 在这个区域内找到与老的初始点对应特征最近的特征, 那么这个特征对应的位置就是新的初始点
        '''
        for i in range(n):
            # 以初始点为中心生成一个正方形mask,
            patch = utils.create_square_mask(F.shape[2], F.shape[3], 
                                             center=init_pts[i].tolist(), 
                                             radius=r2).to(self.device)
            patch_coordinates = torch.nonzero(patch)  # shape [num_points, 2]
            # 拿到mask区域的特征
            F_qi = F[..., patch_coordinates[:, 0], patch_coordinates[:, 1]] # [N, C, num_points] torch.Size([1, 128, 729])
            # 旧初始点的特征
            f_i = self.F0_resized[..., self.init_pts_0[i][0].long(), self.init_pts_0[i][1].long()] # [N, C, 1]
            # 计算mask内每个特征与老的初始点对应特征的距离
            distances = (F_qi - f_i[:, :, None]).abs().mean(1) # [N, num_points] torch.Size([1, 729])
            # 找到距离最小的，也就是老的初始点对应特征最像的特征
            min_index = torch.argmin(distances)
            new_init_pts[i] = patch_coordinates[min_index] # [row, col] 
            
        return new_init_pts
    
    # 切换设备
    def change_device(self, new_device) -> None:
        if new_device != self.device:
            if self.generator.G is not None:
                self.generator.G = self.generator.G.to(new_device)
            self.device = new_device
            self.generator.device = new_device