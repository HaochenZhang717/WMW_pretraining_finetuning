import re
import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras import initializers as tfki
from tensorflow_probability import distributions as tfd
import tensorflow.keras.mixed_precision as prec
from einops import rearrange
from collections import defaultdict

import common



class RSSM(common.Module):
    def __init__(
        self,
        action_free=False,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__()
        self._action_free = action_free
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
            )
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
            )
        return state

    def fill_action_with_zero(self, action):
        # action: [B, action]
        B, D = action.shape[0], action.shape[1]
        if self._action_free:
            return self._cast(tf.zeros([B, 50]))
        else:
            zeros = self._cast(tf.zeros([B, 50 - D]))
            return tf.concat([action, zeros], axis=1)

    @tf.function
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state["stoch"])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state[f"deter"]], -1)

    def get_dist(self, state):
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if is_first.any():
        prev_state, prev_action = tf.nest.map_structure(
            lambda x: tf.einsum("b,b...->b...", 1.0 - is_first.astype(x.dtype), x),
            (prev_state, prev_action),
        )
        prior = self.img_step(prev_state, prev_action, sample)
        x = tf.concat([prior[f"deter"], embed], -1)
        x = self.get("obs_out", tfkl.Dense, self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior[f"deter"], **stats}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, self.fill_action_with_zero(prev_action)], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x, deter = self._cell(x, [prev_state[f"deter"]])
        deter = deter[0]
        x = self.get("img_out", tfkl.Dense, self._hidden)(x)
        x = self.get("img_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer(f"img_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, balance=0.8):
        post_const = tf.nest.map_structure(tf.stop_gradient, post)
        prior_const = tf.nest.map_structure(tf.stop_gradient, prior)
        lhs = tfd.kl_divergence(self.get_dist(post_const), self.get_dist(prior))
        rhs = tfd.kl_divergence(self.get_dist(post), self.get_dist(prior_const))
        return balance * lhs + (1 - balance) * rhs



class TSSM(common.Module):
    def __init__(
        self,
        action_free=False,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,

        deter_type='concat_o',

        n_trans_layers=6,
        n_head=8,
        d_inner=64,
        dropout=0.1,
        dropatt=0.1,
        pre_lnorm=False,
        d_ff_inner=128
    ):
        super().__init__()
        self._action_free = action_free
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std

        self._deter_type=deter_type

        self._n_trans_layers = n_trans_layers
        self._d_model = hidden
        self._n_head = n_head
        self._d_inner = d_inner
        self._dropatt = dropatt
        self._dropout = dropout
        self._pre_lnorm = pre_lnorm
        self._d_ff_inner=d_ff_inner

        self.cfg_trans = {'d_model':hidden,
                          'n_head':n_head,
                          'd_inner':d_inner,
                          'dropout':dropout,
                          'dropatt':dropatt,
                          'pre_lnorm':pre_lnorm,
                          'd_ff_inner':d_ff_inner}

        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def fill_action_with_zero(self, action):
        # action: [B, action]
        B, D = action.shape[:-1], action.shape[-1]
        if self._action_free:
            return self._cast(tf.zeros(list(B) + [50]))
        else:
            zeros = self._cast(tf.zeros(list(B) + [50 - D]))
            return tf.concat([action, zeros], axis=1)

    def _generate_square_subsequent_mask(self, T, H, W):
        N = H * W
        mask = tf.linalg.band_part(tf.ones((T, T)), 0, -1).transpose((1, 0))
        mask = tf.where(mask == 0, -float('1e10'), mask)
        mask = tf.where(mask == 1, float('0.0'), mask)
        mask = tf.repeat(mask, N, axis=0)
        mask = tf.repeat(mask, N, axis=1)
        mask = self._cast(mask)
        return mask

    @tf.function
    def observe(self, prev_stoch, action, sample=True):
        '''
        :param prev_stoch: (B,T,N,C)
        :param action: (B,T,1) in action-free it does not matter
        :param is_first: (B,T,)
        :param state: for TSSM, I think it is trivial
        :return: prior distribution
        '''
        prev_stoch = self._cast(prev_stoch)
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        B, T ,N, C = prev_stoch.shape
        prev_stoch = tf.reshape(prev_stoch, (B, T, N*C))
        x = tf.concat([prev_stoch, self.fill_action_with_zero(action)], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        x = tf.reshape(x, (B, T, -1, 1, 1))

        B, T, D, H, W = x.shape
        attn_mask = self._generate_square_subsequent_mask(T, H, W)

        pos_ips = tf.range(T * H * W)
        pos_ips = tf.cast(pos_ips, prec.global_policy().compute_dtype)
        pos_ips = tf.reshape(pos_ips, (-1, 1))
        pos_embs = self.get("pos_embs_drop", tfkl.Dropout, self._dropout)(positional_embedding(self._hidden, pos_ips))

        x = rearrange(x, 'b t d h w -> (t h w) b d')
        encoder_inp = x + pos_embs
        output = encoder_inp
        output_list = []

        for i in range(self._n_trans_layers):
            output = self.get(f"trans_encoder{i}", TransformerEncoderLayer, **self.cfg_trans)(output, attn_mask=attn_mask)
            output_list.append(output)
        output = tf.stack(output_list, axis=1)  # T, L, B, D
        output = rearrange(output, '(t h w) l b d -> b t l d h w', h=H, w=W) #(B,T,L,D,H,W)
        output = tf.reshape(output, (B, T, self._n_trans_layers, -1)) #(B,T,L,D)
        if self._deter_type == 'concat_o':
            deter = tf.reshape(output, (B, T, -1))
        else:
            deter = output[:, :, -1]
        #to get the shape right, I put an extra linear here
        deter = self.get("deter_out", tfkl.Dense, self._hidden)(deter)
        deter = self._act(deter)
        x_tilde = deter
        x_tilde = self.get("img_out", tfkl.Dense, self._hidden)(x_tilde)
        x_tilde = self.get("img_out_norm", NormLayer, self._norm)(x_tilde)
        x_tilde = self._act(x_tilde)
        stats = self._suff_stats_layer(f"img_dist", x_tilde)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    @tf.function
    def imagine(self, post, traj, context_len=50, temp=None):
        '''
        :param post: post of latent space, here we use discrete distribution
        :param traj: trajectory consisting of action and video, action free does not matter
        :param context_len: the maximum length of input to TSSM
        :param temp: a coefficient for sample
        :return:
        '''
        post_stoch = post['stoch']
        action = traj['action'][:, :post_stoch.shape[1]]
        imag_length = traj['image'].shape[1] - post_stoch.shape[1]
        pred_state = defaultdict(list)
        for t in range(imag_length):
            pred_prior = self.observe(post_stoch[:, -context_len:], action[:, -context_len:], sample=True)
            for k, v in pred_prior.items():
                pred_state[k].append(v[:, -1:])
            post_stoch = tf.concat([post_stoch, pred_prior['stoch'][:, -1:]], axis=1)
            action = tf.concat([action, action[:, -1:]], axis=1)
        for k, v in pred_state.items():
            pred_state[k] = tf.concat(v, axis=1)
        # in action-free pretrain mode, we concatenate trivial action to action list

        return pred_state

    def get_feat(self, state):
        stoch = self._cast(state["stoch"])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state[f"deter"]], -1)

    def get_dist(self, state):
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist


    @tf.function
    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        B = prev_stoch.shape[0]
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, self.fill_action_with_zero(prev_action)], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        x = tf.reshape(x, (B, 1, -1, 1, 1))
        B, T, D, H, W = x.shape
        attn_mask = self._generate_square_subsequent_mask(T, H, W)

        pos_ips = tf.range(T * H * W)
        pos_ips = tf.cast(pos_ips, prec.global_policy().compute_dtype)
        pos_ips = tf.reshape(pos_ips, (-1, 1))
        pos_embs = self.get("pos_embs_drop", tfkl.Dropout, self._dropout)(positional_embedding(self._hidden, pos_ips))

        x = rearrange(x, 'b t d h w -> (t h w) b d')
        encoder_inp = x + pos_embs
        output = encoder_inp
        output_list = []

        for i in range(self._n_trans_layers):
            output = self.get(f"trans_encoder{i}", TransformerEncoderLayer, **self.cfg_trans)(output,
                                                                                              attn_mask=attn_mask)
            output_list.append(output)
        output = tf.stack(output_list, axis=1)  # T, L, B, D
        output = rearrange(output, '(t h w) l b d -> b t l d h w', h=H, w=W)  # (B,T,L,D,H,W)
        output = tf.reshape(output, (B, T, self._n_trans_layers, -1))  # (B,T,L,D)
        if self._deter_type == 'concat_o':
            deter = tf.reshape(output, (B, T, -1))
        else:
            deter = output[:, :, -1]

        # x, deter = self._cell(x, [prev_state[f"deter"]])
        # deter = deter[0]
        x_tilde = deter
        x_tilde = self.get("img_out", tfkl.Dense, self._hidden)(x_tilde)
        x_tilde = self.get("img_out_norm", NormLayer, self._norm)(x_tilde)
        x_tilde = self._act(x_tilde)
        stats = self._suff_stats_layer(f"img_dist", x_tilde)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, balance=0.8):
        post = tf.nest.map_structure(lambda x: x[:, 1:], post)
        post_const = tf.nest.map_structure(tf.stop_gradient, post)
        prior_const = tf.nest.map_structure(tf.stop_gradient, prior)
        lhs = tfd.kl_divergence(self.get_dist(post_const), self.get_dist(prior))
        rhs = tfd.kl_divergence(self.get_dist(post), self.get_dist(prior_const))
        return balance * lhs + (1 - balance) * rhs


class MLP(common.Module):
    def __init__(
        self, shape, layers=[512, 512, 512, 512], act="elu", norm="none", **out
    ):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        x = x.reshape([-1, x.shape[-1]])
        for index, unit in enumerate(self._layers):
            x = self.get(f"dense{index}", tfkl.Dense, unit)(x)
            x = self.get(f"norm{index}", NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + [x.shape[-1]])
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=True, act="tanh", update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, **kwargs)
        if norm:
            self._norm = NormLayer("layer")
        else:
            self._norm = NormLayer("none")

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        parts = self._norm(parts)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class DistLayer(common.Module):
    def __init__(self, shape, dist="mse", outscale=0.1, min_std=0.1, max_std=1.0):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._max_std = max_std
        self._outscale = outscale

    def __call__(self, inputs):
        kw = {}
        if self._outscale == 0.0:
            kw["kernel_initializer"] = tfki.Zeros()
        else:
            kw["kernel_initializer"] = tfki.VarianceScaling(
                self._outscale, "fan_avg", "uniform"
            )
        out = self.get("out", tfkl.Dense, np.prod(self._shape), **kw)(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", tfkl.Dense, np.prod(self._shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == "mse":
            return common.MSEDist(out, len(self._shape), "sum")
        if self._dist == "symlog":
            return common.SymlogDist(out, len(self._shape), "sum")
        if self._dist == "nmse":
            return common.NormalizedMSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._min_std, self._max_std
            std = (hi - lo) * tf.nn.sigmoid(std) + lo
            dist = tfd.Normal(tf.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "trunc_normal":
            lo, hi = self._min_std, self._max_std
            std = (hi - lo) * tf.nn.sigmoid(std) + lo
            dist = tfd.TruncatedNormal(tf.tanh(out), std, -1, 1)
            dist = tfd.Independent(dist, 1)
            dist.minent = np.prod(self._shape) * tfd.Normal(0.99, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "onehot":
            dist = common.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * np.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)


class NormLayer(common.Module, tf.keras.layers.Layer):
    def __init__(self, impl):
        super().__init__()
        self._impl = impl

    def build(self, input_shape):
        if self._impl == "keras":
            self.layer = tfkl.LayerNormalization()
            self.layer.build(input_shape)
        elif self._impl == "layer":
            self.scale = self.add_weight("scale", input_shape[-1], tf.float32, "Ones")
            self.offset = self.add_weight(
                "offset", input_shape[-1], tf.float32, "Zeros"
            )

    def call(self, x):
        if self._impl == "none":
            return x
        elif self._impl == "keras":
            return self.layer(x)
        elif self._impl == "layer":
            mean, var = tf.nn.moments(x, -1, keepdims=True)
            return tf.nn.batch_normalization(
                x, mean, var, self.offset, self.scale, 1e-3
            )
        else:
            raise NotImplementedError(self._impl)


class MLPEncoder(common.Module):
    def __init__(
        self, act="elu", norm="none", layers=[512, 512, 512, 512], batchnorm=False
    ):
        self._act = get_act(act)
        self._layers = layers
        self._norm = norm
        self._batchnorm = batchnorm

    @tf.function
    def __call__(self, x, training=False):
        x = x.astype(prec.global_policy().compute_dtype)
        if self._batchnorm:
            x = self.get(f"batchnorm", tfkl.BatchNormalization)(x, training=training)
        for i, unit in enumerate(self._layers):
            x = self.get(f"dense{i}", tfkl.Dense, unit)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class CNNEncoder(common.Module):
    def __init__(
        self,
        cnn_depth=64,
        cnn_kernels=(4, 4),
        act="elu",
    ):
        self._act = get_act(act)
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

    @tf.function
    def __call__(self, x):
        x = x.astype(prec.global_policy().compute_dtype)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2**i * self._cnn_depth
            x = self.get(f"conv{i}", tfkl.Conv2D, depth, kernel, 1)(x)
            x = self._act(x)
        return x


class CNNDecoder(common.Module):
    def __init__(
        self,
        out_dim,
        cnn_depth=64,
        cnn_kernels=(4, 5),
        act="elu",
    ):
        self._out_dim = out_dim
        self._act = get_act(act)
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

    @tf.function
    def __call__(self, x):
        x = x.astype(prec.global_policy().compute_dtype)

        x = self.get("convin", tfkl.Dense, 2 * 2 * 2 * self._cnn_depth)(x)
        x = tf.reshape(x, [-1, 1, 1, 8 * self._cnn_depth])

        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 1) * self._cnn_depth
            x = self.get(f"conv{i}", tfkl.Conv2DTranspose, depth, kernel, 1)(x)
            x = self._act(x)
        x = self.get("convout", tfkl.Dense, self._out_dim)(x)
        return x


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_inner, dropout, dropatt, pre_lnorm, **kwargs):
        super().__init__(**kwargs)
        self.d_inner = d_inner
        self.n_head = n_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.d_model = d_model

        self.qkv_net = tf.keras.layers.Dense(
            units=3*self.d_inner*self.n_head,
            use_bias=False,
            name='qkv_net'
        )

        self.out_net = tf.keras.layers.Dense(
            units=self.d_model,
            use_bias=False,
            name='out_net'
        )

        self.drop = tf.keras.layers.Dropout(dropout,name='drop')
        self.dropatt = tf.keras.layers.Dropout(dropatt, name='dropatt')
        self.layer_norm = tf.keras.layers.LayerNormalization(name='norm')

        self.scale = 1 / (d_inner ** 0.5)

        self.pre_lnorm = pre_lnorm

    @tf.function
    def call(self, inps, attn_mask=None):
        '''
        :param inps: here it refers q, k, and v. it is self-attention
        :param attn_mask: (T-1,T-1)
        :return:
        '''
        inps = inps.astype(prec.global_policy().compute_dtype)
        T_q, bsz = inps.shape[:2]
        T_k, bsz = inps.shape[:2]

        if self.pre_lnorm:
            w_head_qkv = self.qkv_net(self.layer_norm(inps))
        else:
            w_head_qkv = self.qkv_net(inps)

        w_head_qkv = tf.reshape(w_head_qkv, (T_q, bsz, 3, -1))
        w_head_qkv = tf.transpose(w_head_qkv, (2, 0, 1, 3))
        w_head_q, w_head_k, w_head_v = w_head_qkv[0], w_head_qkv[1], w_head_qkv[2]


        w_head_q = tf.reshape(w_head_q, (T_q, bsz, self.n_head, self.d_inner))
        w_head_k = tf.reshape(w_head_k, (T_k, bsz, self.n_head, self.d_inner))
        w_head_v = tf.reshape(w_head_v, (T_k, bsz, self.n_head, self.d_inner))

        attn_score = tf.einsum('ibnd,jbnd->ijbn', w_head_q, w_head_k) * self.scale#(T-1,T-1,B,n_head)

        #### compute attention probability
        if attn_mask is not None:
            if prec.global_policy().compute_dtype == 'float16':
                attn_score = tf.where(tf.cast(attn_mask[:, :, None, None], dtype=bool), -np.float16('inf'), attn_score)
            elif prec.global_policy().compute_dtype == 'float32':
                attn_score = tf.where(tf.cast(attn_mask[:, :, None, None], dtype=bool), -np.float32('inf'), attn_score)

        attn_prob = tf.nn.softmax(attn_score, axis=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)#(T-1,B,n_head,d_inner)

        # [qlen x bsz x n_head x d_head]
        attn_vec = tf.reshape(attn_vec, (attn_vec.shape[0], attn_vec.shape[1], self.n_head*self.d_inner))

        ##### linear projection
        attn_out = self.out_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(attn_out)

        return output


class PointwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff_inner, dropout, pre_lnorm, act='elu', **kwargs):
        super().__init__(**kwargs)
        self.pre_lnorm = pre_lnorm
        self.d_model = d_model
        self.d_ff_inner = d_ff_inner
        self.dropout = dropout

        self.dense0 =  tf.keras.layers.Dense(
            units=self.d_ff_inner,
            use_bias=False,
            name='dense0'
        )
        self._act = get_act(act)
        self.dense1 =  tf.keras.layers.Dense(
            units=self.d_model,
            use_bias=False,
            name='dense1'
        )
        self.drop = tf.keras.layers.Dropout(dropout,name='drop')
        self.layer_norm = tf.keras.layers.LayerNormalization(name='norm')



    @tf.function
    def call(self, x):
        x = x.astype(prec.global_policy().compute_dtype)
        if self.pre_lnorm:
            x = self.layer_norm(x)
        x = self.dense0(x)
        x = self._act(x)
        x = self.dense1(x)
        x = self.drop(x)
        if not self.pre_lnorm:
            x = self.layer_norm(x)

        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_inner, dropout, dropatt, pre_lnorm, d_ff_inner, **kwargs):
        super().__init__(**kwargs)
        self.mah = MultiheadAttention(d_model, n_head, d_inner, dropout, dropatt, pre_lnorm)
        self.pos_ff = PointwiseFF(d_model, d_ff_inner, dropout, pre_lnorm)

    @tf.function
    def call(self, inpts, attn_mask=None):
        src2 = self.mah(inpts, attn_mask=attn_mask)
        src = inpts + src2
        src2 = self.pos_ff(src)
        src = src + src2
        return src


def get_act(name):
    if name == "none":
        return tf.identity
    if name == "mish":
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)


@tf.function
def positional_embedding(dim, positins):
    inv_freq = 1 / (10000 ** (tf.range(0.0, dim, 2.0) / dim))
    inv_freq = tf.reshape(inv_freq, (1, -1))
    inv_freq = inv_freq.astype(prec.global_policy().compute_dtype)
    positins = positins.astype(prec.global_policy().compute_dtype)
    sinusoid_inp = tf.einsum("ij,jk->ik", positins, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
    return pos_emb[:, None, :]
    # dim=self._hidden
    # positins = pos_ips