import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from copy import deepcopy


class Conv2dSame(nn.Conv2d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# helper method
def sconv2d(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop("padding", 0)
    if isinstance(padding, str):
        if padding.lower() == "same":
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        else:
            # 'valid'
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def padding_arg(default, padding_same=False):
    return "SAME" if padding_same else default


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Default args for PyTorch BN impl
BN_MOMENTUM_DEFAULT = 0.1
BN_EPS_DEFAULT = 1e-5


def round_channels(channels, depth_multiplier=1.0, depth_divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    if not depth_multiplier:
        return channels

    channels *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_channels = max(
        int(channels + depth_divisor / 2) // depth_divisor * depth_divisor, min_depth
    )
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += depth_divisor
    return new_channels


def swish(x):
    return x * torch.sigmoid(x)


def hard_swish(x):
    return x * F.relu6(x + 3.0) / 6.0


def hard_sigmoid(x):
    return F.relu6(x + 3.0) / 6.0


def drop_connect(inputs, training=False, drop_connect_rate=0.0):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device
    )
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        act_fn=F.relu,
        bn_momentum=BN_MOMENTUM_DEFAULT,
        bn_eps=BN_EPS_DEFAULT,
        folded_bn=False,
        padding_same=False,
    ):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        self.act_fn = act_fn
        padding = padding_arg(get_padding(kernel_size, stride), padding_same)

        self.conv = sconv2d(
            in_chs, out_chs, kernel_size, stride=stride, padding=padding, bias=folded_bn
        )
        self.bn1 = (
            None
            if folded_bn
            else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)
        )

    def forward(self, x):
        x = self.conv(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        act_fn=F.relu,
        noskip=False,
        pw_act=False,
        se_ratio=0.0,
        se_gate_fn=torch.sigmoid,
        bn_momentum=BN_MOMENTUM_DEFAULT,
        bn_eps=BN_EPS_DEFAULT,
        folded_bn=False,
        padding_same=False,
        drop_connect_rate=0.0,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_se = se_ratio is not None and se_ratio > 0.0
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        dw_padding = padding_arg(kernel_size // 2, padding_same)
        pw_padding = padding_arg(0, padding_same)

        self.conv_dw = sconv2d(
            in_chs,
            in_chs,
            kernel_size,
            stride=stride,
            padding=dw_padding,
            groups=in_chs,
            bias=folded_bn,
        )
        self.bn1 = (
            None
            if folded_bn
            else nn.BatchNorm2d(in_chs, momentum=bn_momentum, eps=bn_eps)
        )

        if self.has_se:
            self.se = SqueezeExcite(
                in_chs,
                reduce_chs=max(1, int(in_chs * se_ratio)),
                act_fn=act_fn,
                gate_fn=se_gate_fn,
            )

        self.conv_pw = sconv2d(in_chs, out_chs, 1, padding=pw_padding, bias=folded_bn)
        self.bn2 = (
            None
            if folded_bn
            else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)
        )

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=torch.sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool bad for NVIDIA AMP performance
        # tensor.view + mean bad for ONNX export (produces mess of gather ops that break TensorRT)
        # x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block w/ optional SE"""

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        act_fn=F.relu,
        exp_ratio=1.0,
        noskip=False,
        se_ratio=0.0,
        se_reduce_mid=False,
        se_gate_fn=torch.sigmoid,
        bn_momentum=BN_MOMENTUM_DEFAULT,
        bn_eps=BN_EPS_DEFAULT,
        folded_bn=False,
        padding_same=False,
        drop_connect_rate=0.0,
    ):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        dw_padding = padding_arg(kernel_size // 2, padding_same)
        pw_padding = padding_arg(0, padding_same)

        # Point-wise expansion
        self.conv_pw = sconv2d(in_chs, mid_chs, 1, padding=pw_padding, bias=folded_bn)
        self.bn1 = (
            None
            if folded_bn
            else nn.BatchNorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)
        )

        # Depth-wise convolution
        self.conv_dw = sconv2d(
            mid_chs,
            mid_chs,
            kernel_size,
            padding=dw_padding,
            stride=stride,
            groups=mid_chs,
            bias=folded_bn,
        )
        self.bn2 = (
            None
            if folded_bn
            else nn.BatchNorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)
        )

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs,
                reduce_chs=max(1, int(se_base_chs * se_ratio)),
                act_fn=act_fn,
                gate_fn=se_gate_fn,
            )

        # Point-wise linear projection
        self.conv_pwl = sconv2d(mid_chs, out_chs, 1, padding=pw_padding, bias=folded_bn)
        self.bn3 = (
            None
            if folded_bn
            else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)
        )

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        if self.bn3 is not None:
            x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants

        return x


class EfficientNetBuilder:
    """Build Trunk Blocks for Efficient/Mobile Networks
    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py
    """

    def __init__(
        self,
        channel_multiplier=1.0,
        channel_divisor=8,
        channel_min=None,
        drop_connect_rate=0.0,
        act_fn=None,
        se_gate_fn=torch.sigmoid,
        se_reduce_mid=False,
        bn_momentum=BN_MOMENTUM_DEFAULT,
        bn_eps=BN_EPS_DEFAULT,
        folded_bn=False,
        padding_same=False,
    ):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.drop_connect_rate = drop_connect_rate
        self.act_fn = act_fn
        self.se_gate_fn = se_gate_fn
        self.se_reduce_mid = se_reduce_mid
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.folded_bn = folded_bn
        self.padding_same = padding_same

        # updated during build
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return round_channels(
            chs, self.channel_multiplier, self.channel_divisor, self.channel_min
        )

    def _make_block(self, ba):
        bt = ba.pop("block_type")
        ba["in_chs"] = self.in_chs
        ba["out_chs"] = self._round_channels(ba["out_chs"])
        ba["bn_momentum"] = self.bn_momentum
        ba["bn_eps"] = self.bn_eps
        ba["folded_bn"] = self.folded_bn
        ba["padding_same"] = self.padding_same
        # block act fn overrides the model default
        ba["act_fn"] = ba["act_fn"] if ba["act_fn"] is not None else self.act_fn
        if bt == "ir":
            ba["drop_connect_rate"] = (
                self.drop_connect_rate * self.block_idx / self.block_count
            )
            ba["se_gate_fn"] = self.se_gate_fn
            ba["se_reduce_mid"] = self.se_reduce_mid
            block = InvertedResidual(**ba)
        elif bt == "ds" or bt == "dsa":
            ba["drop_connect_rate"] = (
                self.drop_connect_rate * self.block_idx / self.block_count
            )
            block = DepthwiseSeparableConv(**ba)
        elif bt == "cn":
            block = ConvBnAct(**ba)
        else:
            assert False, "Uknkown block type (%s) while building model." % bt
        self.in_chs = ba["out_chs"]  # update in_chs for arg of next block
        return block

    def _make_stack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for bi, ba in enumerate(stack_args):
            if bi >= 1:
                # only the first block in any stack/stage can have a stride > 1
                ba["stride"] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1  # incr global idx (across all stacks)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list delimits stacks (stages),
                inner list contains args defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        # outer list of arch_args defines the stacks ('stages' by some conventions)
        for stack_idx, stack in enumerate(block_args):
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


def _decode_block_str(block_str, depth_multiplier=1.0):
    """Decode block definition string
    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.
    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act,
      ca = Cascade3x3, and possibly more)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    a - activation fn ('re', 'r6', or 'hs')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split("_")
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op.startswith("a"):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == "re":
                value = F.relu
            elif v == "r6":
                value = F.relu6
            elif v == "hs":
                value = hard_swish
            else:
                continue
            options[key] = value
        elif op == "noskip":
            noskip = True
        else:
            # all numeric options
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_fn is None, the model default (passed to model init) will be used
    act_fn = options["a"] if "a" in options else None

    num_repeat = int(options["r"])
    # each type of block has different valid arguments, fill accordingly
    if block_type == "ir":
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options["k"]),
            out_chs=int(options["c"]),
            exp_ratio=float(options["e"]),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=int(options["s"]),
            act_fn=act_fn,
            noskip=noskip,
        )
        if "g" in options:
            block_args["pw_group"] = options["g"]
            if options["g"] > 1:
                block_args["shuffle_type"] = "mid"
    elif block_type == "ds" or block_type == "dsa":
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options["k"]),
            out_chs=int(options["c"]),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=int(options["s"]),
            act_fn=act_fn,
            noskip=block_type == "dsa" or noskip,
            pw_act=block_type == "dsa",
        )
    elif block_type == "cn":
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options["k"]),
            out_chs=int(options["c"]),
            stride=int(options["s"]),
            act_fn=act_fn,
        )
    else:
        assert False, "Unknown block type (%s)" % block_type

    # return a list of block args expanded by num_repeat and
    # scaled by depth_multiplier
    num_repeat = int(math.ceil(num_repeat * depth_multiplier))
    return [deepcopy(block_args) for _ in range(num_repeat)]


def decode_arch_def(arch_def, depth_multiplier=1.0):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            stack_args.extend(_decode_block_str(block_str, depth_multiplier))
        arch_args.append(stack_args)
    return arch_args


def _initialize_weight_goog(m):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(0)  # fan-out
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def _initialize_weight_default(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="linear")


class GenEfficientNet(nn.Module):
    """Generic Efficent Networks
    An implementation of mobile optimized networks that covers:
      * EfficientNet
      * MobileNet V1, V2, and V3
      * MNASNet A1, B1, and small
      * FBNet C
      * ChamNet
      * Single-Path NAS Pixel1
    """

    def __init__(
        self,
        block_args,
        num_classes=1000,
        in_chans=1,
        stem_size=32,
        num_features=1280,
        channel_multiplier=1.0,
        channel_divisor=8,
        channel_min=None,
        bn_momentum=BN_MOMENTUM_DEFAULT,
        bn_eps=BN_EPS_DEFAULT,
        drop_rate=0.0,
        drop_connect_rate=0.0,
        act_fn=F.relu,
        se_gate_fn=torch.sigmoid,
        se_reduce_mid=False,
        head_conv="default",
        weight_init="goog",
        folded_bn=False,
        padding_same=False,
    ):
        super(GenEfficientNet, self).__init__()
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_classes = num_classes

        stem_size = round_channels(
            stem_size, channel_multiplier, channel_divisor, channel_min
        )
        self.conv_stem = sconv2d(
            in_chans,
            stem_size,
            3,
            padding=padding_arg(1, padding_same),
            stride=2,
            bias=folded_bn,
        )
        self.bn1 = (
            None
            if folded_bn
            else nn.BatchNorm2d(stem_size, momentum=bn_momentum, eps=bn_eps)
        )
        in_chs = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier,
            channel_divisor,
            channel_min,
            drop_connect_rate,
            act_fn,
            se_gate_fn,
            se_reduce_mid,
            bn_momentum,
            bn_eps,
            folded_bn,
            padding_same,
        )
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        if not head_conv or head_conv == "none":
            self.efficient_head = False
            self.conv_head = None
            assert in_chs == num_features
        else:
            self.efficient_head = head_conv == "efficient"
            self.conv_head = sconv2d(
                in_chs,
                num_features,
                1,
                padding=padding_arg(0, padding_same),
                bias=folded_bn and not self.efficient_head,
            )
            self.bn2 = (
                None
                if (folded_bn or self.efficient_head)
                else nn.BatchNorm2d(num_features, momentum=bn_momentum, eps=bn_eps)
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if weight_init == "goog":
                _initialize_weight_goog(m)
            else:
                _initialize_weight_default(m)

        # self.scnas = ScNas1d(3, 12, 3, 6)

    def features(self, x):
        x = self.conv_stem(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)
        x = self.blocks(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        if self.efficient_head:
            x = F.adaptive_avg_pool2d(x, 1)
            x = self.conv_head(x)
            # no BN
            x = self.act_fn(x)
        else:
            if self.conv_head is not None:
                x = self.conv_head(x)
                if self.bn2 is not None:
                    x = self.bn2(x)
                x = self.act_fn(x)
            # x = F.adaptive_avg_pool2d(x, 1)
            x = self.avgpool(x)
        return x

    def forward(self, x):
        _min = 13152  # 5 percentile of training set
        _max = 13663  # 95 percentile of training set
        x = x.clamp(_min, _max)
        x = (x - _min) / (_max - _min)
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)

        x = self.classifier(x)
        return x


_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
_BN_EPS_TF_DEFAULT = 1e-3


def _resolve_bn_params(kwargs):
    # NOTE kwargs passed as dict intentionally
    bn_momentum_default = BN_MOMENTUM_DEFAULT
    bn_eps_default = BN_EPS_DEFAULT
    bn_tf = kwargs.pop("bn_tf", False)
    if bn_tf:
        bn_momentum_default = _BN_MOMENTUM_TF_DEFAULT
        bn_eps_default = _BN_EPS_TF_DEFAULT
    bn_momentum = kwargs.pop("bn_momentum", None)
    bn_eps = kwargs.pop("bn_eps", None)
    if bn_momentum is None:
        bn_momentum = bn_momentum_default
    if bn_eps is None:
        bn_eps = bn_eps_default
    return bn_momentum, bn_eps


def _gen_efficientnet(
    channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs
):
    """Creates an EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenEfficientNet(
        decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        channel_divisor=8,
        channel_min=None,
        num_features=round_channels(1280, channel_multiplier, 8, None),
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        act_fn=swish,
        **kwargs
    )
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    """EfficientNet-B0 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
    """
    model = _gen_efficientnet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    return model


def efficientnet_b1(pretrained=False, **kwargs):
    """EfficientNet-B1"""
    model = _gen_efficientnet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    return model


def efficientnet_b2(pretrained=False, **kwargs):
    """EfficientNet-B2"""
    model = _gen_efficientnet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    return model


def efficientnet_b3(pretrained=False, **kwargs):
    """EfficientNet-B3"""
    # NOTE for train, drop_rate should be 0.3
    model = _gen_efficientnet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    return model


def efficientnet_b4(pretrained=False, **kwargs):
    """EfficientNet-B4"""
    model = _gen_efficientnet(channel_multiplier=1.4, depth_multiplier=1.8, **kwargs)
    return model


def efficientnet_b5(pretrained=False, **kwargs):
    """EfficientNet-B4"""
    model = _gen_efficientnet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    return model


if __name__ == "__main__":
    loader = efficientnet_b5(num_classes=1080, in_chans=1)
    print(loader)
    input_ = torch.rand([4, 1, 512, 512])
    o_s = loader(input_)
    print(o_s.shape)