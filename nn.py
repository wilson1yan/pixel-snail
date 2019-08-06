from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn


''' utilities for shifting the image around, efficient alternative to masking convolutions '''
def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0) )                           # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down :
            self.down_shift = partial(down_shift, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = partial(right_shift, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu

        if skip_connection != 0 :
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None :
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * torch.sigmoid(b)
        return og_x + c3

class causal_attention(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        canvas_size = obs_dim[1] * obs_dim[2]
        self.canvas_size = canvas_size
        self.register_buffer('causal_mask', self._create_mask(canvas_size))

    def _create_mask(self, canvas_size):
        causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
        for i in range(canvas_size):
            causal_mask[i, :i] = 1.
        causal_mask = np.expand_dims(causal_mask, axis=0)
        return torch.FloatTensor(causal_mask)

    def forward(self, key, mixin, query):
        qs, ks, ms = query.shape, key.shape, mixin.shape
        query = query.permute(0, 2, 3, 1).contiguous().view(qs[0], self.canvas_size, qs[1])
        key = key.permute(0, 2, 3, 1).contiguous().view(ks[0], self.canvas_size, ks[1])
        dot = torch.matmul(query, key.transpose(1, 2)) - (1. - self.causal_mask) * 1e10
        dot = dot - torch.max(dot, axis=-1, keedim=True)[0]
        causal_exp_dot = torch.exp(dot / np.sqrt(ks[-1]).astype(np.float32)) * self.causal_mask
        causal_probs = causal_exp_dot / (torch.sum(causal_exp_dot, axis=-1, keeydim=True) + 1e-6)
        mixin = mixin.permute(0, 2, 3, 1).contiguous().view(ms[0], self.canvas_size, ms[1])
        mixed = torch.matmul(causal_probs, mixin)
        return mixed.view(qs[0], qs[2], qs[3], ms[1]).permute(0, 3, 1, 2).contiguous()

###############################################################################
#                           utils                                             #
###############################################################################

def log_prob_from_logits(x, *, dim=1):
    m, _ = torch.max(x, dim, keepdim=True)
    return x - m - torch.logsumexp(x - m, dim, keepdim=True)


def logistic_logpdf(w, inv_stdv, log_scale):
    return w - log_scale - 2 * F.softplus(w)


def discretized_mix_logistic_loss(x, l, nr_mix, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = x.shape  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = l.shape  # predicted distribution, e.g. (B,32,32,100)
    # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :nr_mix, :, :]
    l = torch.reshape(l[:, nr_mix:, :, :], xs[:2] + [nr_mix * 3] + xs[2:])
    means = l[:, :, :nr_mix, :, :]
    log_scales = torch.clamp(l[:, :, nr_mix:2 * nr_mix, :, :], -7)

    # Linear - conditional color channels if nr_color_channels == 3. No need if
    # grayscale (only one color channel)
    coeffs = torch.tanh(l[:, :, 2 * nr_mix:3 * nr_mix, :, :])
    # here and below: getting the means and adjusting them based on preceding sub-pixels
    x = torch.reshape(x, xs[:2] + [1] + xs[2:])
    x = x.repeat(1, 1, nr_mix, 1, 1)
    m2 = torch.reshape(means[:, 1, :, :, :] + coeffs[:, 0, :, :, :]
                       * x[:, 0, :, :, :], [xs[0], 1, nr_mix, xs[2], xs[3]])
    m3 = torch.reshape(means[:, 2, :, :, :] + coeffs[:, 1, :, :, :] * x[:, 0, :, :, :] +
                       coeffs[:, 2, :, :, :] * x[:, 1, :, :, :], [xs[0], 1, nr_mix, xs[2], xs[3]])
    means = torch.cat([torch.reshape(means[:, 0, :, :, :], [xs[0], 1, nr_mix, xs[2], xs[3]]), m2, m3], dim=1)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
    log_pdf_mid = logistic_logpdf(mid_in, inv_stdv, log_scales)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = torch.sum(log_probs, 1) + \
        log_prob_from_logits(logit_probs, dim=1)
    if sum_all:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -torch.sum(log_sum_exp(log_probs), [1, 2])


def sample_from_discretized_mix_logistic(l, nr_mix, device='cpu'):
    ls = l.shape  # (B, 100, 32, 32)
    xs = [ls[0]] + [3] + ls[2:]  # (B, 3, 32, 32)
    # unpack parameters
    logit_probs = l[:, :nr_mix, :, :]
    l = torch.reshape(l[:, nr_mix:, :, :], xs[:2] + [nr_mix * 3] + xs[2:])
    # sample mixture indicator from softmax
    minval = 1e-5
    maxval = 1. - 1e-5
    sel_inds = torch.argmax(logit_probs - torch.log(-torch.log(
        torch_random_uniform(logit_probs.shape, minval=minval, maxval=maxval).to(device))), dim=1)
    sel = torch_one_hot(sel_inds, depth=nr_mix, axis=1,
                        dtype=torch.float32, device=device)
    sel = torch.reshape(sel, xs[:1] + [1, nr_mix] + xs[2:])
    # select logistic parameters
    means = torch.sum(l[:, :, :nr_mix, :, :] * sel, 2)
    log_scales = torch.clamp(
        torch.sum(l[:, :, nr_mix:2 * nr_mix, :, :] * sel, 2), min=-7.)
    log_scales -= temperature

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch_random_uniform(means.shape, minval=minval,
                             maxval=maxval).to(device)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    coeffs = torch.sum(torch.tanh(
        l[:, :, 2 * nr_mix:3 * nr_mix, :, :]) * sel, 2)
    x0 = torch.clamp(x[:, 0, :, :], min=-1., max=1.)
    x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :]
                     * x0, min=-1., max=1.)
    x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] *
                     x0 + coeffs[:, 2, :, :] * x1, min=-1., max=1.)
    temp_s = [ls[0]] + [1] + ls[2:]
    return torch.cat([torch.reshape(x0, temp_s), torch.reshape(x1, temp_s), torch.reshape(x2, temp_s)], 1)
