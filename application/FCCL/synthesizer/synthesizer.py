import time
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from abc import ABC
from torchvision import transforms
from coala.distributed.distributed import CPU
from kornia import augmentation
from .utils import DeepInversionHook, ImagePool, fomaml_grad, kldiv, reptile_grad, reset_l0_fun


class GlobalSynthesizer(ABC):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='data/save_syn', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, distributed=False, lr_z=0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                 is_maml=1,
                 device=CPU):
        super(GlobalSynthesizer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.is_maml = is_maml
        self.device = device

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.to(device).train()
        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        if self.is_maml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])

        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, targets=None):
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)

        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(self.device)
        z.requires_grad = True
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0]  # sort for better visualization
        targets = targets.to(self.device)

        fast_generator = self.generator.clone()
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        for it in range(self.iterations):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs)  # crop and normalize
            t_out = self.teacher(inputs_aug)
            
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.to(self.device)

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets)
            if self.adv > 0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()  # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.cpu()  # mem, cpu

            optimizer.zero_grad()
            loss.backward()

            if self.is_maml:
                if it == 0:
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()

            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.is_maml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)

        self.data_pool.add(best_inputs)  # add a batch of data


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
