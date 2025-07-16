'''
    两个支路：
        检测支路
        视频分类支路
        源头检测支路
    CenterNet: 对所有视频帧检测，
    视频分类采用tsa
    分割网络对源头检测
'''

from torch import nn
import torch.nn.functional as TNF

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
import torch.nn.init as init
from torch.nn.init import normal_, constant_
import senet
# from torch.utils.tensorboard import SummaryWriter
# import cv2

# import TRNmodule

BN_MOMENTUM = 0.1

def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    
class Deconv(nn.Module):
    def __init__(self, inplanes=256):
        super(Deconv, self).__init__()
        self.deconv_with_bias = False
        self.inplanes = inplanes

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        deconv_x = self.deconv_layers(x)
        return deconv_x

def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = TNF.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class  DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward_corr(self, inputs, kernels):
        assert inputs.size(0) == len(kernels), 'batch size not equal for Corr calculate'
        features = []
        h, w = inputs.shape[2:]
        for i, kernel in enumerate(kernels):
            if kernel.size(-1) == 0:
                features.append(inputs[i].unsqueeze(0))
            else:
                kernel = self.conv_kernel(kernel)
                input = self.conv_input(inputs[i].unsqueeze(0))
                feature = conv2d_dw_group(input, kernel)
                feature = TNF.interpolate(feature, (h, w), mode='bilinear', align_corners=True)

                features.append(feature)
        features = torch.cat(features, dim=0)
        return features

    def forward(self, kernel, input):
        features = self.forward_corr(kernel, input)
        out = self.head(features)
        return out

class TSN(nn.Module):
    def __init__(self, num_class=2, num_segments=8, modality="RGB",
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec == True:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        self.conv_fuse1 = nn.Conv2d(2048, 256, 1, 1, bias=True)
        self.deconv = Deconv(256)

        self.seg_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, 1, 1)
        )

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        #[B,2048,7,7]
        #input = input.view((-1, sample_len) + input.size()[-2:])
        feature_maps = self.base_model(input)
        #[B,256,7,7]
        out = TNF.relu(self.conv_fuse1(feature_maps), inplace=True)

        #[B,256,56,56]
        out = self.deconv(out)
        #[B,1,56,56]
        seg_out = self.seg_layer(out)
        #[B,1,256,256]
        seg_out = TNF.upsample(seg_out, size=(input.shape[2], input.shape[3]), mode='bilinear')
        seg_out = seg_out.sigmoid_()
        return seg_out

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type in ['TRN','TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'resnext' in base_model:
            self.base_model = getattr(senet, base_model)(num_classes=1000, pretrained='imagenet')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        transpose = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.ConvTranspose2d):
                transpose.extend(m.parameters())
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': transpose, 'lr_mult': 1, 'decay_mult': 0,
             'name': 'conv transpose'}
        ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        #return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
        #                                           GroupRandomHorizontalFlip()])
        rrc = RandomResizedCrop(size=224, scale=(0.9, 1), ratio=(3./4., 4./3.))
        rp = RandomPerspective(anglex=3, angley=3, anglez=3, shear=3)
        # Improve generalization
        rhf = RandomHorizontalFlip(p=0.5)
        # Deal with dirts, ants, or spiders on the camera lense
        re = RandomErasing(p=0.5, scale=(0.003, 0.01), ratio=(0.3, 3.3), value=0)
        cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1), gamma=0.3)

        return torchvision.transforms.Compose([cj, rrc, rp, rhf])
        #return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .9,]),
        #                                           GroupRandomHorizontalFlip()])
if __name__ == '__main__':
    model = TSN(2,
                8,
                'RGB',
                base_model='se_resnext50_32x4d',
                consensus_type='TRNmultiscale',
                img_feature_dim=256, print_spec=False)
    model = model.cuda()
    model.eval()
    img = torch.randn(2,3,256,256).cuda()
    seg = model(img)
    print(seg.shape)
