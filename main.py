import torch, torchvision
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataloader import Datasat
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import math, os, copy
from einops import rearrange
from Spectral_Recon import *
from Spatial_Recon import *


# +--------------------------------
#       some global params
# +--------------------------------

ms_channels = 4
hs_channels = 102

"""
    Define super-resolution network in both domains
"""


class S2TD(nn.Module):
    def __init__(self):
        super(S2TD, self).__init__()
        self.SpectralPriorModule = SpectralPrior(hs_channels, hs_channels)
        self.SpectralDeModule = SpecDataIntegrity()
        self.SpatialPriorModule = SpatialPrior(hs_channels, hs_channels)
        self.SpatialDeModule = SpatDataIntegrity()


    def forward(self, LRHS, HRMS, LRHS_noisy, time_emb):
        # stage1--Spatial
        H0_Spat = torch.zeros_like(LRHS)
        X0_Spat = self.SpatialPriorModule(LRHS_noisy, LRHS, H0_Spat, time_emb, HRMS)
        Y_Spat = nn.functional.interpolate(LRHS, scale_factor=0.25)

        # stage1--Spectral
        H0_Spec = torch.zeros_like(LRHS)
        X0_Spec= self.SpectralPriorModule(LRHS_noisy, LRHS, H0_Spat, time_emb, LRHS)
        Y_Spec= HRMS

        # stage2--Spatial
        H1_Spat = self.SpatialDeModule(X0_Spat, H0_Spat, Y_Spat)
        X1_Spat = self.SpatialPriorModule(LRHS_noisy, X0_Spec, H1_Spat, time_emb, HRMS)

        # stage2--Spectral
        H1_Spec = self.SpectralDeModule(X0_Spec, H0_Spec, Y_Spec)
        X1_Spec = self.SpectralPriorModule(LRHS_noisy, X1_Spat, H1_Spec, time_emb, LRHS)

        # stage3--Spatial
        H2_Spat = self.SpatialDeModule(X1_Spat, H1_Spat, Y_Spat)
        X2_Spat = self.SpatialPriorModule(LRHS_noisy, X1_Spec, H2_Spat, time_emb, HRMS)

        # stage3--Spectral
        H2_Spec = self.SpectralDeModule(X1_Spec, H1_Spec, Y_Spec)
        X2_Spec = self.SpectralPriorModule(LRHS_noisy, X1_Spat, H2_Spec, time_emb, LRHS)

        return X2_Spec, X2_Spat


class L1Charbonnierloss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1Charbonnierloss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""


class Diffusion(nn.Module):
    def __init__(self, model, device, channels=128):
        super(Diffusion, self).__init__()
        self.channels = channels
        self.model = model.to(device)
        self.device = device


    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-3):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.register_buffer('pred_coef3', to_torch(np.sqrt(1. / 1 - alphas_cumprod)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, condition1, condition2):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        pred_spectral, pred_spatial = self.model(condition1, condition2, x, noise_level)
        posterior_mean = (
                self.posterior_mean_coef1[t] * pred_spectral.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, LRHS, HRMS, img_spec, img_spat, t):
        mean_spec, log_variance_spec = self.p_mean_variance(x=img_spec, t=t, condition1=LRHS, condition2=HRMS)
        noise_spec = torch.randn_like(img_spec) if t > 0 else torch.zeros_like(img_spec)

        mean_spat, log_variance_spat = self.p_mean_variance(x=img_spat, t=t, condition1=LRHS, condition2=HRMS)
        noise_spat = torch.randn_like(img_spat) if t > 0 else torch.zeros_like(img_spat)
        return mean_spec + noise_spec * (0.5 * log_variance_spec).exp(), mean_spat + noise_spat * (0.5 * log_variance_spat).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, HRMS, LRHS):
        img_spec = torch.rand_like(LRHS, device=device)
        img_spat = torch.rand_like(LRHS, device=device)
        for i in reversed(range(0, self.num_timesteps)):
            img_spec, img_spat = self.p_sample(LRHS, HRMS, img_spec, img_spat, i)
        return img_spec, img_spat

    # Compute loss to train the model
    def p_losses(self, HRMS, LRHS): # x_in=lrHS

        t = np.random.randint(1, self.num_timesteps + 1)
        x_start = LRHS
        b, c, h, w = x_start.shape
        noise = torch.randn_like(x_start).to(x_start.device)

        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha ** 2).sqrt() * noise
        pred_spectral, pred_spatial = self.model(LRHS, HRMS, x_noisy, sqrt_alpha)

        return pred_spectral, pred_spatial

    def forward(self, HRMS, LRHS):
        return self.p_losses(HRMS, LRHS)


# Class to train & test desired model
class DualDiffusion():
    def __init__(self, device, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path, load=False,
                 out_channel=hs_channels, inner_channel=32, norm_groups=8,
                 channel_mults=(1, 2, 4, 8), res_blocks=2, dropout=0, lr=1e-5, distributed=False):
        super(DualDiffusion, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path

        model = S2TD().to(device)
        self.sr = Diffusion(model, device, out_channel)

        # Apply weight initialization & set loss & set noise schedule
        self.sr.apply(self.weights_init_orthogonal)
        self.sr.set_loss(loss_type)
        self.sr.set_new_noise_schedule(schedule_opt)

        self.l1loss = nn.L1Loss().to(device)
        self.l1charloss = L1Charbonnierloss().to(device)

        if distributed:
            assert torch.cuda.is_available()
            self.model = nn.DataParallel(self.sr)

        self.optimizer = torch.optim.Adam(self.sr.parameters(), lr=lr)
        params = sum(p.numel() for p in self.sr.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):
        best_loss = 100

        for i in tqdm(range(epoch)):
            train_loss = 0
            self.sr.train()

            for _, (hrms, lrhs, gt, lrms) in enumerate(self.dataloader):
                    # Initial imgs are high-resolution
                    gt = gt.to(device)
                    hrms = hrms.to(device)

                    lrms = lrms.to(device)

                    b, c, h, w = hrms.shape
                    self.optimizer.zero_grad()

                    # 扩散模型分布损失

                    pred_spectral, pred_spatial = self.sr(lrms, lrms_morec)
                    loss1 = self.l1charloss(pred_spatial, gt)
                    loss2 = self.l1charloss(pred_spectral, gt)
                    loss3 = self.l1loss(pred_spectral, pred_spatial)

                    loss = loss1 + loss2 + loss3
                    loss = loss.sum()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item() * b
            print("epoch:{}, loss:{:.6f}".format(i, loss / len(self.dataloader)))
            # print("loss:{:.6f}".format(loss / len(self.dataloader)))

            if i % 1 == 0:
                if train_loss <= best_loss:
                    best_loss = train_loss
                    # Save model weight
                    self.save(self.save_path)


            if (i + 1) % verbose == 0:
                    test_imgs = next(iter(self.testloader))

                    test_imgs[0] = test_imgs[0].to(device)
                    test_imgs[2] = test_imgs[2].to(device)
                    test_imgs[3] = test_imgs[3].to(device)

                    ratio = hs_channels // ms_channels
                    hrms_morec = test_imgs[0].repeat(1, ratio, 1, 1)
                    hrms_morec = torch.cat([hrms_morec, test_imgs[0][:, 0:2, :, :]], dim=1)

                    test_imgs[1] = torch.nn.functional.interpolate(test_imgs[1], scale_factor=0.25)
                    test_imgs[1] = torch.nn.functional.interpolate(test_imgs[1], scale_factor=4).to(device)  #lrhs

                    train_loss = train_loss / len(self.dataloader)
                    print(f'Epoch: {i + 1} / loss:{train_loss:.6f}')

                    img_spec, img_spat = self.test(test_imgs[0], test_imgs[1])



    def test(self, HRMS, LRHS):
            self.sr.eval()
            with torch.no_grad():
                img_spec, img_spat = self.sr.super_resolution(HRMS, LRHS)
            self.sr.train()
            return img_spec, img_spat

    def save(self, save_path):
        network = self.sr
        if isinstance(self.sr, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.sr
        if isinstance(self.sr, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=False)
        print("Spectral Model loaded successfully")

if __name__ == "__main__":
    train_batch = 2
    test_batch = 1

    traindata = Datasat('train')
    testdata = Datasat('test')
    dataloader = DataLoader(dataset=traindata, batch_size=train_batch, shuffle=True)
    testloader = DataLoader(dataset=testdata, batch_size=test_batch, shuffle=False)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if cuda else "cpu")
    schedule_opt = {'schedule': 'cosine', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    diff = DualDiffusion(device, loss_type='l1',
              dataloader=dataloader, testloader=testloader, schedule_opt=schedule_opt,
              save_path='/media/xd132/USER/LS/interpret-blindSR/model/pavia.pt',
              load_path='/media/xd132/USER/LS/interpret-blindSR/model/pavia.pt',
              load=False, inner_channel=16,
              norm_groups=16, channel_mults=(1, 2, 4, 8), dropout=0.2, res_blocks=2, lr=1e-4, distributed=False)

