import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import h5py
from load_vel import overthrust_model
from generator import generator
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
sfmt=ticker.ScalarFormatter(useMathText=True) 
sfmt.set_powerlimits((0, 0))
import matplotlib

class Sample(object):
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
        self.build_model(args)
        
    def build_model(self, args):

        m0, m, self.dm, spacing, shape, origin = overthrust_model(args.vel_dir)
        self.extent = np.array([0., self.dm.shape[2]*spacing[0], 
            self.dm.shape[3]*spacing[1], 0.])/1.0e3
        self.dm = self.dm.to(self.device) 
        self.load(args, os.path.join(args.checkpoint_dir, args.experiment))
        self.burn_in_index = 52

    def load(self, args, checkpoint_dir):

        log_to_load = os.path.join(checkpoint_dir, 'training-logs.pt')
        assert os.path.isfile(log_to_load)

        if args.cuda == 0:
            training_logs = torch.load(log_to_load, map_location='cpu')
        else:
            training_logs = torch.load(log_to_load)
        print(' [*] Samples loaded')
        self.net_loss_log = training_logs['net_loss_log']
        self.model_loss_log = training_logs['model_loss_log']
        self.samples = training_logs['samples']
        assert len(self.samples) > self.burn_in_index
        
    def test(self, args):

        
        fig = plt.figure("profile", dpi=200, figsize=(7, 2.5))
        plt.imshow(self.dm[0, 0, :, :].t().cpu().numpy(), vmin=-3.0/100.0, vmax=3.0/100.0, 
            aspect=1, extent=self.extent, cmap="seismic", alpha=0.6, interpolation="kaiser")
        plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.title("True model - "  + r"$\delta \mathbf{m}$");
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "dm.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        self.samples = np.array(self.samples)
        self.samples = np.transpose(self.samples.reshape((-1, self.dm.shape[2], self.dm.shape[3])), 
            (0, 2, 1))
        for j in range(self.samples.shape[0]):
            self.samples[j, :, :] = self.model_topMute(self.samples[j, :, :])

        samples_mean = np.mean(self.samples[self.burn_in_index:, :, :], axis=0)
        samples_std = np.std(self.samples[self.burn_in_index:, :, :], axis=0)

        if not os.path.exists(os.path.join(args.sample_dir, args.experiment, "Gzs")):
            os.makedirs(os.path.join(args.sample_dir, args.experiment, "Gzs"))

        idxs = np.random.choice(self.samples[self.burn_in_index:, :, :].shape[0], 5, replace=False)
        for i in idxs:
            fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 2.5))
            plt.imshow(self.samples[self.burn_in_index + i], vmin=-3.0/100.0, vmax=3.0/100.0, aspect=1, \
                extent=self.extent, cmap="seismic", alpha=0.6, interpolation="kaiser")
            plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$\mathbf{g}(\mathbf{z},\widehat{{\mathbf{w}}}$" + r"$_{{{}}})$".format(i));
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                "Gzs", "Gz" + str(i) + ".png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            fig = plt.figure("G(z_0) - G(z_i)", dpi=100, figsize=(7, 2.5))
            plt.imshow(self.samples[self.burn_in_index + i] - self.samples[self.burn_in_index], \
                vmin=-2e-2, vmax=2e-2, aspect=1, extent=self.extent, cmap="twilight_shifted", interpolation="kaiser")
            plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$\mathbf{g}(\mathbf{z},\widehat{{\mathbf{w}}}$" + r"$_{{{}}}) - $".format(i) + 
                r"$\mathbf{g}(\mathbf{z},\widehat{{\mathbf{w}}}_{{0}})$");   
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                "Gzs", "Gz_" + str(i) + "-Gz0.png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            fig = plt.figure("G(z_0) -mean", dpi=100, figsize=(7, 2.5))
            plt.imshow(self.samples[self.burn_in_index + i] - samples_mean, \
                vmin=-2e-2, vmax=2e-2, aspect=1, extent=self.extent, cmap="twilight_shifted", interpolation="kaiser")
            plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$\mathbf{g}(\mathbf{z},\widehat{{\mathbf{w}}}$" + r"$_{{{}}}) - $".format(i) + 
                r"$\delta \widehat { \mathbf{m}}$");   
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                "Gzs", "Gz_" + str(i) + "-mean.png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        fig = plt.figure("mean of G(z) over random z's", dpi=100, figsize=(7, 2.5))
        plt.imshow(samples_mean, vmin=-3.0/100.0, vmax=3.0/100.0, aspect=1, extent=self.extent, cmap="seismic", 
            alpha=0.6, interpolation="kaiser")
        plt.title(r"$\delta \widehat { \mathbf{m}} $" + " - mean of " + 
            r"$\mathbf{g}(\mathbf{z},\widehat{{\mathbf{w}}}_j)$"  + "'s" + 
            r"$, \ \widehat{{\mathbf{w}}}_j \sim p_{post} ( \mathbf{w} |  \left \{ \mathbf{d}_{i}, \mathbf{q}_{i} \right \}_{i=1}^N, \mathbf{z} )$")
        plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "Gz-mean.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        
        x_loc = [334, 64]
        y_loc = [65, 79]
        fig = plt.figure("std of G(z) over random z's", dpi=100, figsize=(7, 2.5))
        plt.imshow(samples_std, vmin=0., vmax=9e-3, aspect=1, extent=self.extent, cmap="OrRd", 
            interpolation="kaiser")
        plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
        plt.plot(x_loc[0]*0.025, y_loc[0]*0.025, marker="o", ms=10, alpha=0.9, c="#00b4ba", 
            markerfacecolor="None", markeredgewidth=1.2)
        plt.plot(x_loc[1]*0.025, y_loc[1]*0.025, marker="o", ms=10, alpha=0.9, c="#00b4ba",
            markerfacecolor="None", markeredgewidth=1.2)
        plt.plot(x_loc[0]*0.025, y_loc[0]*0.025, marker="o", ms=10, alpha=0.2, c="None",
            markerfacecolor="#00b4ba", markeredgewidth=.01)
        plt.plot(x_loc[1]*0.025, y_loc[1]*0.025, marker="o", ms=10, alpha=0.2, c="None",
            markerfacecolor="#00b4ba", markeredgewidth=.01)
        plt.title("Point-wise standard deviation of " + r"$\mathbf{g}(\mathbf{z},\widehat{{\mathbf{w}}}_j)$" + "'s")
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "Gz-std.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        norm_fac = 0.0098
        fig = plt.figure("profile", dpi=200, figsize=(7, 2.5))
        plt.imshow(self.dm[0, 0, :, :].t().cpu().numpy(), vmin=-3.0/100.0, vmax=3.0/100.0, 
            aspect=1, extent=self.extent, cmap="seismic", alpha=0.3, interpolation="kaiser")
        horiz_loz = [50, 150, 250, 350]
        for loc in horiz_loz:
            plt.plot(samples_std[:, loc]/norm_fac + loc*.025, 
                np.linspace(0., 3.025, samples_std.shape[0]),  
                color="#0a9c00", lw=1.4, alpha=0.7);
            plt.plot(np.zeros(self.dm.shape[3]) + loc*.025, 
                np.linspace(0., 3.025, samples_std.shape[0]), color="k", lw=1.4, alpha=0.5);
        plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt);
        plt.title("Point-wise standard deviation vertical profiles");
        plt.xlabel("Horizontal distance (km)"); plt.ylabel("Depth (km)")
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "overlaid-std.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def sample_prior(self, args):

        samples_to_load = os.path.join('./checkpoint', 'prior_samples.pt')
        if os.path.isfile(samples_to_load):
            self.prior_samples = torch.load(samples_to_load)['prior_samples']
            print(' [*] Prior samples loaded')
        else:
            print(' [*] Computing samples from the prior')
            self.prior_samples = []
            self.z = torch.randn((1, 3, 512, 128), device=self.device, requires_grad=False)
            for j in tqdm(range(5000)):
                self.G = generator(
                            self.dm.size(),
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
                self.prior_samples.append(self.G(self.z).detach().cpu().numpy())

            torch.save({'prior_samples': self.prior_samples}, os.path.join('./checkpoint',
                'prior_samples.pt'))
            print(' [*] Prior samples saved')

        self.prior_samples = np.array(self.prior_samples)
        self.prior_samples = np.transpose(self.prior_samples.reshape((-1, self.dm.shape[2], self.dm.shape[3])), 
            (0, 2, 1))

        samples_mean = np.mean(self.prior_samples, axis=0)
        samples_std = np.std(self.prior_samples, axis=0)

        if not os.path.exists(os.path.join(args.sample_dir, args.experiment, "Prior")):
            os.makedirs(os.path.join(args.sample_dir, args.experiment, "Prior"))

        idxs = np.random.choice(1000, 5, replace=False)
        for i in idxs:
            fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 2.5))
            plt.imshow(self.prior_samples[i], vmin=-20.0/100.0, vmax=20.0/100.0, aspect=1, \
                extent=self.extent, cmap="seismic", alpha=0.6, interpolation="kaiser")
            plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$\mathbf{g}(\mathbf{z},\mathbf{w}_0)$" 
                + r"$, \ \mathbf{w}_0 \sim p_{prior} ( \mathbf{w} )$")
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                "Prior", "Gz" + str(i) + ".png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        fig = plt.figure("mean of G(z) over random z's", dpi=100, figsize=(7, 2.5))
        plt.imshow(samples_mean, vmin=np.min(samples_mean), vmax=-np.min(samples_mean), 
            aspect=1, extent=self.extent, cmap="seismic", 
            alpha=0.6, interpolation="kaiser")
        plt.title("Mean of " + r"$\mathbf{g}(\mathbf{z},\mathbf{w}_i)$"  + "'s" + 
            r"$, \ \mathbf{w}_i \sim p_{prior} ( \mathbf{w} )$")
        plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "Prior-mean.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        fig = plt.figure("std of G(z) over random z's", dpi=100, figsize=(7, 2.5))
        plt.imshow(samples_std, vmin=np.min(samples_std), vmax=np.max(samples_std), 
            aspect=1, extent=self.extent, cmap="OrRd", 
            interpolation="kaiser")
        plt.title("Point-wise standard deviation of " + r"$\mathbf{g}(\mathbf{z},\mathbf{w}_i)$"  + "'s" )
        plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "Prior-std.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        assert len(self.samples) > 0
        self.samples = np.array(self.samples)
        self.samples = np.transpose(self.samples.reshape((-1, self.dm.shape[2], self.dm.shape[3])), 
            (0, 2, 1))
        for j in range(self.samples.shape[0]):
            self.samples[j, :, :] = self.model_topMute(self.samples[j, :, :])
        self.samples = self.samples[self.burn_in_index:, :, :]
        x_loc = [334, 64]
        y_loc = [65, 79]
        for ix, iy in zip(x_loc, y_loc):
            hist_init = []
            hist_trained = []
            for i in range(self.prior_samples.shape[0]):
                hist_init.append(self.prior_samples[i, iy, ix])
            for i in range(self.samples.shape[0]):
                hist_trained.append(self.samples[i, iy, ix])
            fig = plt.figure("hist", dpi=100, figsize=(7, 2))
            n, bins, _ = plt.hist(np.array(hist_init), bins=np.linspace(-0.10, 0.10, 100), 
                density=False, label="prior", color="#ff8800", alpha=0.5,  histtype='bar')
            plt.hist(np.array(hist_trained), 12, density=True, label="posterior", 
                color="#00b4ba", alpha=0.8, histtype='bar')
            plt.title("Point-wise histogram at (" + "{0:.2f}".format(ix*.025) + 
                " km, " + "{0:.2f}".format(iy*.025) + " km)");
            # plt.vlines(self.dm[0, 0, ix, iy], 0, 200, lw=0.8, label=r"$\delta \mathbf{m}$")
            plt.xlabel("Perturbation");
            plt.legend()
            plt.grid()
            plt.xlim([-0.10, 0.10])
            plt.ylim([0, 125])
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "histogram-at-" + 
                "{}".format(ix) + "x" + "{}".format(iy) + ".png"), format="png", 
                bbox_inches="tight", dpi=300)
            plt.close(fig)

            # plt.stem(bins[:-1],n/10)


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