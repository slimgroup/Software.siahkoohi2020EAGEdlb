import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from wave_solver import *
import h5py
from load_vel import overthrust_model
from precondSGLD import pSGLD
from generator import generator
from tensorboardX import SummaryWriter
import matplotlib.ticker as ticker
from tqdm import tqdm
sfmt=ticker.ScalarFormatter(useMathText=True) 
sfmt.set_powerlimits((0, 0))

class LearnedImaging(object):
    def __init__(self, args):

        if torch.cuda.is_available() and args.cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print(' [*] GPU is available')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        if not os.path.exists(os.path.join(args.vel_dir, 'data/')):
            os.makedirs(os.path.join(args.vel_dir, 'data/'))

        data_path = os.path.join(args.vel_dir, 'data/nonlinear-seq-shots.hdf5')
        if not os.path.isfile(data_path):
            os.system("wget https://www.dropbox.com/s/04a3xblk0634mm4/nonlinear-seq-shots.hdf5 -O" 
                + data_path)            
        self.y = h5py.File(data_path, 'r')["data"][...]
        
        # Seq. src: lin_err_std = 0.4849208975730595; lin_err_mean: 0.00013945832330958545
        # Sim. src: lin_err_std = 0.49033560630591055; lin_err_mean: -2.642165903863294e-05
        lin_err_std = 0.49033560630591055
        self.sigma_squared = args.eta + lin_err_std**2
        self.num_exps = np.prod(self.y.shape[0])
        self.epoch = torch.Tensor([args.epoch])
        self.build_model(args)
        
    def build_model(self, args):

        m0, m, self.dm, spacing, shape, origin = overthrust_model(args.vel_dir)
        self.extent = np.array([0., self.dm.shape[2]*spacing[0], 
            self.dm.shape[3]*spacing[1], 0.])/1.0e3
        self.x = self.dm.to(self.device)
        self.wave_solver = wave_solver(self.y, shape, origin, spacing, m0, self.dm, 
            noise=args.eta, device=self.device, sequential=False)
        self.net_loss_log = []
        self.model_loss_log = []
        self.z = torch.randn((1, 3, 512, 128), device=self.device, requires_grad=False)
        self.G = generator(
                    self.x.size(),
                    num_input_channels=3, num_output_channels=1, 
                    num_channels_down = [16, 32, 256],
                    num_channels_up   = [16, 32, 256],
                    num_channels_skip = [0, 0, 0],
                    upsample_mode = 'bicubic',
                    need1x1_up = True,
                    filter_size_down=5, 
                    filter_size_up=3,
                    filter_skip_size = 1,
                    need_sigmoid=False, 
                    need_bias=True, 
                    pad='reflection', 
                    act_fun='LeakyReLU').to(self.device)

        self.l2_loss = torch.nn.MSELoss().to(self.device)
        self.optim = pSGLD([{'params': self.G.parameters()}], 
            float(args.lr), 
            weight_decay=args.weight_decay*np.prod(self.y.shape)/(args.eta))

    def train(self, args):

        self.writer = SummaryWriter('logs/' + args.experiment)
        start_time = time.time()
        self.current_epoch = torch.Tensor([0])
        self.samples = []

        As = []
        ys = []
        print(' [*] Creating sim. source experiemts and associated born operators')
        for ne in tqdm(range(self.num_exps)):
            As.append(self.wave_solver.create_operators())
            ys.append(self.wave_solver.mix_data())
            ys[-1] = torch.from_numpy(ys[-1])
            ys[-1] = ys[-1].to(self.device)

        while (self.current_epoch < self.epoch)[0]:

            exp_idx = np.random.choice(self.num_exps, 1)[0]
            A = As[exp_idx]
            y = ys[exp_idx]

            x_est = self.G(self.z)
            pred = A(x_est)
            net_loss = self.l2_loss(pred.reshape(-1), y.reshape(-1))*np.prod(self.y.shape)/(self.sigma_squared)/2.0
            model_loss = self.l2_loss(x_est.reshape(-1), self.x.reshape(-1))

            grad_theta = torch.autograd.grad(net_loss, self.G.parameters(), create_graph=False)
            for param, grad in zip(self.G.parameters(), grad_theta):
                param.grad = grad
            self.optim.step()

            self.net_loss_log.append(net_loss.detach())
            self.model_loss_log.append(model_loss.detach())
            self.writer.add_scalar('net_loss', net_loss, self.current_epoch)
            self.writer.add_scalar('model_loss', model_loss, self.current_epoch)
            print(("Iteration: [%d/%d] | time: %4.2f | neural net loss: %f | model loss: %4.8f" % \
                (self.current_epoch+1, self.epoch, time.time() - start_time, net_loss, model_loss)))

            if (self.current_epoch > self.num_exps)[0] and torch.fmod(self.current_epoch, 
                args.sample_freq)==0:
                self.samples.append(x_est.detach().cpu().numpy())

            if torch.fmod(self.current_epoch, args.save_freq) == 0 \
                or self.current_epoch == self.epoch - 1:
                self.save(os.path.join(args.checkpoint_dir, args.experiment), self.current_epoch)
                self.test(args)
            self.current_epoch += 1

    def save(self, checkpoint_dir, current_epoch):

        torch.save({'net_loss_log': self.net_loss_log,
            'samples': self.samples,
            'model_loss_log': self.model_loss_log}, os.path.join(checkpoint_dir,
            'training-logs.pt'))

        torch.save({'model_state_dict': self.G.state_dict(),
            'z': self.z}, os.path.join(checkpoint_dir,
            'checkpoint.pth'))

    def test(self, args):
        if len(self.samples)>0:
            fig = plt.figure("G(z_0)", dpi=300, figsize=(7, 2.5))
            plt.imshow(self.model_topMute(np.transpose(self.samples[-1].reshape((self.dm.shape[2], 
                self.dm.shape[3])))), vmin=-3.0/90.0, vmax=5.0/90.0, aspect=1, extent=self.extent)
            plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "Gz0_" + 
                str(self.current_epoch.item()) + ".png"), format="png", 
                bbox_inches="tight", dpi=300)
            plt.close(fig)

        fig = plt.figure("training logs - net", dpi=300, figsize=(7, 2.5))
        plt.semilogy(self.net_loss_log); plt.title(r"$\|\|y_{i}-A_{i}g(z_{i},w)\|\|_2^2$")
        plt.grid()
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "training-loss.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        fig = plt.figure("training logs - model", dpi=300, figsize=(7, 2.5))
        plt.semilogy(self.model_loss_log); plt.title(r"$\|\| \delta {m} - g(z_{i},w)\|\|_2^2$")
        plt.grid()
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "model-loss.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def model_topMute(self, image, mute_end=20, length=1):

        mute_start = mute_end - length
        damp = np.zeros([image.shape[0]])
        damp[0:mute_start-1] = 0.
        damp[mute_end:] = 1.
        taper_length = mute_end - mute_start + 1
        taper = (1. + np.sin((np.pi/2.0*np.array(range(0,taper_length-1)))/(taper_length - 1)))/2.
        damp[mute_start:mute_end] = taper
        for j in range(0, image.shape[1]):
            image[:,j] = image[:,j]*damp
        return image
