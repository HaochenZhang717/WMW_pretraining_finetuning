import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common


class Agent(common.Module):
    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = WorldModel(config, obs_space, self.act_space, self.tfstep)

    @tf.function
    def train(self, data):
        metrics = {}
        state, outputs, mets = self.wm.train(data)
        metrics.update(mets)

        return state, metrics

    @tf.function
    def train_mae(self, data):
        metrics = {}
        mets = self.wm.train_mae(data)
        metrics.update(mets)
        return metrics


    @tf.function
    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        report["openl_image"] = self.wm.video_pred(data)
        return report


class WorldModel(common.Module):
    def __init__(self, config, obs_space, act_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.act_space = act_space

        # RSSM
        self.tssm = common.TSSM(**config.tssm)
        # self.feat_dim = config.rssm.deter + config.rssm.stoch * config.rssm.discrete

        # MAE
        self.mae_encoder, self.mae_decoder = common.mae_factory(**config.mae)

        # ViT for latent dynamics model
        self.wm_vit_encoder, self.wm_vit_decoder = common.flat_vit_factory(
            **config.wm_flat_vit
        )

        # Optimizers
        self.model_opt = common.Optimizer("model", **config.model_opt)
        self.mae_opt = common.Optimizer("mae", **config.mae_opt)

        # ImageNet stats
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406])
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225])

        self.step = 0

    def train(self, data):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data)
        modules = [
            self.tssm,
            self.wm_vit_encoder,
            self.wm_vit_decoder,
        ]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def train_mae(self, data):
        with tf.GradientTape() as mae_tape:
            mae_loss, metrics = self.loss_mae(data, training=True)
        modules = [
            self.mae_encoder,
            self.mae_decoder,
        ]
        metrics.update(self.mae_opt(mae_tape, mae_loss, modules))
        return metrics

    def loss(self, data):
        data = self.preprocess(data)
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        likes, losses, metrics = {}, {}, {}

        # Forward without masking
        m = 0.0
        latent, mask, _ = self.mae_encoder.forward_encoder(videos, m, T)
        feature = latent
        data["feature"] = tf.stop_gradient(feature.astype(tf.float32))
        # data["feature"] = data["feature"].reshape([B, T, feature.shape[-2], feature.shape[-1]])
        # data["feature"] = data["feature"][:, 1:].reshape([B*(T-1), feature.shape[-2], feature.shape[-1]])
        # Detach features
        feature = tf.stop_gradient(feature)

        # ViT encoder with average pooling
        ## Move [CLS] to last position
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        post = self.wm_vit_encoder.forward_encoder(feature, B, T)
        feat = self.tssm.get_feat(post)
        # TSSM forward
        prior = self.tssm.observe(post["stoch"][:, :-1], data["action"][:, :-1], sample=True)
        kl_loss = kl_value = self.tssm.kl_loss(post, prior, self.config.wmkl_balance)
        losses["kl"] = tf.clip_by_value(
            kl_loss * self.config.wmkl.scale, self.config.wmkl_minloss, 100.0
        ).mean()
        # Feature reconstruction loss
        feat = tf.reshape(feat, [B * T, 1, -1])
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)
        dist = common.MSEDist(tf.cast(feature_pred, tf.float32), 1, "sum")
        like = tf.cast(dist.log_prob(data["feature"]), tf.float32)
        likes["feature"] = like
        losses["feature"] = -like.mean()

        # Summation and log metrics
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        # feat is prior feat contains oth deter and stoch;
        # likes is logliklihood of feature reconstruction;
        # kl_value is similarity between prior and post;
        outs = dict(
         feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.tssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.tssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def loss_mae(self, data, training=False):
        data = self.preprocess(data)
        key = "image"
        videos = data[key]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        losses, metrics = {}, {}

        # MAE forward
        m = self.config.mask_ratio
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, 1)


        if self.config.mae.reward_pred:
            decoder_pred, reward_pred = self.mae_decoder.forward_decoder(
                latent, ids_restore
            )
            # Reward prediction loss
            reward_pred = tf.reshape(reward_pred, [B, T, 1])
            reward = tf.reshape(data["reward"], [B, T, 1])
            reward_loss = self.mae_decoder.forward_reward_loss(reward, reward_pred)
            losses["mae_reward"] = reward_loss
        else:
            decoder_pred = self.mae_decoder.forward_decoder(latent, ids_restore)

        # Image reconstruction loss
        decoder_loss = self.mae_decoder.forward_loss(videos, decoder_pred, mask)
        losses[key] = decoder_loss

        # Summation and log metrics
        mae_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        return mae_loss, metrics

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:  # image
                value = self.standardize(value.astype(dtype) / 255.0)
            obs[key] = value
        return obs

    @tf.function
    def standardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = (x - mean) / std
        return x

    @tf.function
    def destandardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = x * std + mean
        return x

    @tf.function
    def video_pred(self, data):
        data = {k: v[:6] for k, v in data.items()}
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])

        # Autoencoder reconstruction
        m = 0.0 if self.config.mae.early_conv else self.config.mask_ratio
        recon_latent, recon_mask, recon_ids_restore = self.mae_encoder.forward_encoder(
            videos, m, T
        )
        recon_model = self.mae_decoder.forward_decoder(recon_latent, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon_model = recon_model[0]  # first element is decoded one
        recon_model = tf.cast(recon_model, tf.float32)
        recon_model = self.mae_decoder.unpatchify(recon_model[: B * T])

        recon_model = tf.cast(
            self.destandardize(recon_model.reshape([B, T, H, W, C])), tf.float32
        )

        # Latent dynamics model prediction
        # 1: Extract MAE representations
        m = 0.0
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, T)
        feature = tf.stop_gradient(latent)

        # 2: Reconstructions from conditioning frames
        # 2-1: Process through ViT encoder
        ## Move [CLS] to last position for positional embedding
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        wm_latent = self.wm_vit_encoder.forward_encoder(feature, bitch_size=B, video_length=T)
        # embed = wm_latent.mean(1).reshape([B, T, wm_latent.shape[-1]])

        # 2-2: Process these through TSSM
        states = self.tssm.observe(
            wm_latent['stoch'][:6, :5], data["action"][:6, :5], sample=True
        )
        feat = self.tssm.get_feat(states)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, -1])

        # 2-3: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 2-4 Process these through MAE decoder
        recon_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, :5].reshape(
            [b * t, -1]
        )
        recon = self.mae_decoder.forward_decoder(feature_pred, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon = recon[0]
        recon = tf.cast(recon, tf.float32)
        recon = self.mae_decoder.unpatchify(recon[: b * t])
        recon = tf.reshape(
            recon, [b, t, recon.shape[1], recon.shape[2], recon.shape[3]]
        )
        recon = self.destandardize(recon)

        # 3: Open-loop prediction
        # 3-1: Process through RSSM to obtain prior
        prior = self.tssm.imagine(tf.nest.map_structure(lambda x: x[:6, :5], wm_latent), data)
        feat = self.tssm.get_feat(prior)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, -1])

        # 3-2: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 3-3: Process these through MAE decoder
        openl_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, 5:].reshape(
            [b * t, -1]
        )
        openl = self.mae_decoder.forward_decoder(feature_pred, openl_ids_restore)
        if self.config.mae.reward_pred:
            openl = openl[0]
        openl = tf.cast(openl, tf.float32)
        openl = self.mae_decoder.unpatchify(openl[: b * t])
        openl = tf.reshape(
            openl, [b, t, openl.shape[1], openl.shape[2], openl.shape[3]]
        )
        openl = self.destandardize(openl)

        # Concatenate across timesteps
        model = tf.concat([recon, openl], 1)
        truth = tf.cast(
            self.destandardize(videos.reshape([B, T, H, W, C])[:6]), tf.float32
        )
        video = tf.concat([truth, recon_model, model], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ReconstructModel(common.Module):
    def __init__(self, config, obs_space, act_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.act_space = act_space

        # RSSM
        # self.tssm = common.TSSM(**config.tssm)
        self.feat_dim = config.rssm.deter + config.rssm.stoch * config.rssm.discrete

        # MAE
        self.mae_encoder, self.mae_decoder = common.mae_factory(**config.mae)

        # ViT for latent dynamics model
        self.wm_vit_encoder, self.wm_vit_decoder = common.flat_vit_factory(
            **config.wm_flat_vit
        )

        # Optimizers
        self.model_opt = common.Optimizer("model", **config.model_opt)
        self.mae_opt = common.Optimizer("mae", **config.mae_opt)

        # ImageNet stats
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406])
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225])

        self.step = 0

    def train(self, data):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data)
        modules = [
            self.tssm,
            self.wm_vit_encoder,
            self.wm_vit_decoder,
        ]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def train_mae(self, data):
        with tf.GradientTape() as mae_tape:
            mae_loss, metrics = self.loss_mae(data, training=True)
        modules = [
            self.mae_encoder,
            self.mae_decoder,
        ]
        metrics.update(self.mae_opt(mae_tape, mae_loss, modules))
        return metrics

    def loss(self, data):
        data = self.preprocess(data)
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        likes, losses, metrics = {}, {}, {}

        # Forward without masking
        m = 0.0
        latent, mask, _ = self.mae_encoder.forward_encoder(videos, m, T)
        feature = latent
        data["feature"] = tf.stop_gradient(feature.astype(tf.float32))
        data["feature"] = data["feature"].reshape([B, T, feature.shape[-2], feature.shape[-1]])
        data["feature"] = data["feature"][:, 1:].reshape([B*(T-1), feature.shape[-2], feature.shape[-1]])
        # Detach features
        feature = tf.stop_gradient(feature)

        # ViT encoder with average pooling
        ## Move [CLS] to last position
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        post = self.wm_vit_encoder.forward_encoder(feature, B, T)
        # TSSM forward
        prior = self.tssm.observe(post["stoch"][:, :-1], data["action"][:, :-1], sample=True)
        feat = self.tssm.get_feat(prior)
        kl_loss = kl_value = self.tssm.kl_loss(post, prior, self.config.wmkl_balance)
        losses["kl"] = tf.clip_by_value(
            kl_loss * self.config.wmkl.scale, self.config.wmkl_minloss, 100.0
        ).mean()

        # Non-image losses
        # dists = {}
        # for name, head in self.heads.items():
        #     grad_head = name in self.config.grad_heads
        #     inp = feat if grad_head else tf.stop_gradient(feat)
        #     out = head(inp)
        #     out = out if isinstance(out, dict) else {name: out}
        #     dists.update(out)
        # for key, dist in dists.items():
        #     like = tf.cast(dist.log_prob(data[key]), tf.float32)
        #     likes[key] = like
        #     losses[key] = -like.mean()

        # Feature reconstruction loss
        feat = tf.reshape(feat, [B * (T-1), 1, -1])
        # print("================================")
        # print(feat.shape)
        # print("================================")
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)
        dist = common.MSEDist(tf.cast(feature_pred, tf.float32), 1, "sum")
        like = tf.cast(dist.log_prob(data["feature"]), tf.float32)
        likes["feature"] = like
        losses["feature"] = -like.mean()

        # Summation and log metrics
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        # feat is prior feat contains oth deter and stoch;
        # likes is logliklihood of feature reconstruction;
        # kl_value is similarity between prior and post;
        outs = dict(
         feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.tssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.tssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def loss_mae(self, data, training=False):
        data = self.preprocess(data)
        key = "image"
        videos = data[key]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        losses, metrics = {}, {}

        # MAE forward
        m = self.config.mask_ratio
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, 1)


        if self.config.mae.reward_pred:
            decoder_pred, reward_pred = self.mae_decoder.forward_decoder(
                latent, ids_restore
            )
            # Reward prediction loss
            reward_pred = tf.reshape(reward_pred, [B, T, 1])
            reward = tf.reshape(data["reward"], [B, T, 1])
            reward_loss = self.mae_decoder.forward_reward_loss(reward, reward_pred)
            losses["mae_reward"] = reward_loss
        else:
            decoder_pred = self.mae_decoder.forward_decoder(latent, ids_restore)

        # Image reconstruction loss
        decoder_loss = self.mae_decoder.forward_loss(videos, decoder_pred, mask)
        losses[key] = decoder_loss

        # Summation and log metrics
        mae_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        return mae_loss, metrics

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:  # image
                value = self.standardize(value.astype(dtype) / 255.0)
            obs[key] = value
        return obs

    @tf.function
    def standardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = (x - mean) / std
        return x

    @tf.function
    def destandardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = x * std + mean
        return x

    @tf.function
    def video_pred(self, data):
        data = {k: v[:6] for k, v in data.items()}
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])

        # Autoencoder reconstruction
        m = 0.0 if self.config.mae.early_conv else self.config.mask_ratio
        recon_latent, recon_mask, recon_ids_restore = self.mae_encoder.forward_encoder(
            videos, m, T
        )
        recon_model = self.mae_decoder.forward_decoder(recon_latent, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon_model = recon_model[0]  # first element is decoded one
        recon_model = tf.cast(recon_model, tf.float32)
        recon_model = self.mae_decoder.unpatchify(recon_model[: B * T])

        recon_model = tf.cast(
            self.destandardize(recon_model.reshape([B, T, H, W, C])), tf.float32
        )

        # Latent dynamics model prediction
        # 1: Extract MAE representations
        m = 0.0
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, T)
        feature = tf.stop_gradient(latent)

        # 2: Reconstructions from conditioning frames
        # 2-1: Process through ViT encoder
        ## Move [CLS] to last position for positional embedding
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        wm_latent = self.wm_vit_encoder.forward_encoder(feature, bitch_size=B, video_length=T)
        # embed = wm_latent.mean(1).reshape([B, T, wm_latent.shape[-1]])

        # 2-2: Process these through RSSM
        states = self.tssm.observe(
            wm_latent['stoch'][:6, :5], data["action"][:6, :5], sample=True
        )
        feat = self.tssm.get_feat(states)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, -1])

        # 2-3: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 2-4 Process these through MAE decoder
        recon_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, :5].reshape(
            [b * t, -1]
        )
        recon = self.mae_decoder.forward_decoder(feature_pred, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon = recon[0]
        recon = tf.cast(recon, tf.float32)
        recon = self.mae_decoder.unpatchify(recon[: b * t])
        recon = tf.reshape(
            recon, [b, t, recon.shape[1], recon.shape[2], recon.shape[3]]
        )
        recon = self.destandardize(recon)

        # 3: Open-loop prediction
        # 3-1: Process through RSSM to obtain prior
        prior = self.tssm.imagine(tf.nest.map_structure(lambda x: x[:6, :5], wm_latent), data)
        feat = self.tssm.get_feat(prior)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, -1])

        # 3-2: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 3-3: Process these through MAE decoder
        openl_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, 5:].reshape(
            [b * t, -1]
        )
        openl = self.mae_decoder.forward_decoder(feature_pred, openl_ids_restore)
        if self.config.mae.reward_pred:
            openl = openl[0]
        openl = tf.cast(openl, tf.float32)
        openl = self.mae_decoder.unpatchify(openl[: b * t])
        openl = tf.reshape(
            openl, [b, t, openl.shape[1], openl.shape[2], openl.shape[3]]
        )
        openl = self.destandardize(openl)

        # Concatenate across timesteps
        model = tf.concat([recon, openl], 1)
        truth = tf.cast(
            self.destandardize(videos.reshape([B, T, H, W, C])[:6]), tf.float32
        )
        video = tf.concat([truth, recon_model, model], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
