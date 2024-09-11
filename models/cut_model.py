import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import itertools
from . import losses
import math
from models.SRC import SRC_Loss
def tensor2pdf(x):
    b, c, h, w = x.size()
    n = w * h - 1
    x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
    sigma_square = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 1e-4
    pi = torch.tensor(math.pi)
    gau_pdf = (-x_minus_mu_square / (2 * sigma_square)).exp() / (2 * pi * sigma_square).sqrt()

    return gau_pdf
class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=False, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,7,11,15', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
        # parser.add_argument('--netH1', type=str, default='H1')
        # parser.add_argument('--netH2', type=str, default='H2')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
        parser.add_argument('--patch_nums', type=float, default=256,
                            help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
        parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
        parser.add_argument('--learned_attn', action='store_true', help='use the learnable attention map')
        parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=False, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]


        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.normalization = losses.Normalization(self.device)
            self.netPre = losses.VGG16().to(self.device)
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionStyle = losses.StyleLoss().to(self.device)# gram
            self.criterionFeature = losses.PerceptualLoss().to(self.device)
            self.criterionSpatial = losses.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    opt.learned_attn, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)

            self.normalization = losses.Normalization(self.device)
            self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]

            self.criterionR = []
            for nce_layer in self.nce_layers:
                self.criterionR.append(SRC_Loss(opt).to(self.device))
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)


    def optimize_parameters(self):
        # forward
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)
        self.idt_B = self.netG(self.real_B)


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        fake_B_feat = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
        real_A_feat = self.netG(self.real_A, self.nce_layers, encode_only=True)
        fake_B_pool, sample_ids = self.netF(fake_B_feat, self.opt.num_patches, None)
        real_A_pool, _ = self.netF(real_A_feat, self.opt.num_patches, sample_ids)

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            idt_B_feat = self.netG(self.idt_B, self.nce_layers, encode_only=True)
            if self.opt.flip_equivariance and self.flipped_for_equivariance:
                idt_B_feat = [torch.flip(fq, [3]) for fq in idt_B_feat]
            real_B_feat = self.netG(self.real_B, self.nce_layers, encode_only=True)

            idt_B_pool, _ = self.netF(idt_B_feat, self.opt.num_patches, sample_ids)
            real_B_pool, _ = self.netF(real_B_feat, self.opt.num_patches, sample_ids)


        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(real_A_pool, fake_B_pool)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:

            self.loss_G_s_idt_B = self.Spatial_Loss(self.netPre, norm_real_B, norm_fake_idt_B, None)*10
            self.loss_NCE_Y = self.calculate_NCE_loss(real_B_pool, idt_B_pool)

        norm_real_A = tensor2pdf(self.real_A)
        norm_fake_B = tensor2pdf(self.fake_B)
        self.loss_H_s = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None) * 10

        self.loss_G = self.loss_G_GAN + self.loss_NCE + self.loss_SRC + self.loss_H_s  + self.loss_G_s_idt_B

        return self.loss_G
    

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q_pool = tgt
        feat_k_pool = src
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers


    def calculate_R_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_SRC_loss = 0.0

        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionR):
            loss_SRC = crit(f_q, f_k)
            total_SRC_loss += loss_SRC

        return total_SRC_loss / n_layers