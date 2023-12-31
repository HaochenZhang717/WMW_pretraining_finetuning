import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["somethingv2"]).parse(known_only=True)
    config = common.Config(configs["defaults"])

    for name in parsed.configs:
        config = config.update(configs[name])
    # config = config.update(configs["somethingv2"])

    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
    print("Loading Logdir", load_logdir)

    import tensorflow as tf

    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        import tensorflow.keras.mixed_precision as prec
        prec.set_global_policy("mixed_float16")

    # train_replay = common.ReplayWithoutAction(
    #     logdir / "train_episodes",
    #     load_directory=load_logdir / "train_episodes",
    #     **config.replay
    # )

    train_replay = common.ReplayWithoutAction(
        logdir,
        load_directory=load_logdir,
        **config.replay
    )



    step = common.Counter(train_replay.stats["total_steps"])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_log = common.Every(config.log_every)
    should_save = common.Every(config.eval_every)

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = common.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera
            )
            env = common.NormalizeAction(env)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = common.MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = common.NormalizeAction(env)
        elif suite == "rlbench":
            env = common.RLBench(
                task,
                config.render_size,
                config.action_repeat,
            )
            env = common.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, config.time_limit)
        return env


    print("Create envs.")
    env = make_env("train")
    act_space, obs_space = env.act_space, env.obs_space

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    mae_train_dataset = iter(train_replay.dataset(**config.mae_dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_mae = agnt.train_mae
    train_agent = common.CarryOverState(agnt.train)

    train_mae(next(mae_train_dataset))
    train_agent(next(train_dataset))

    # agnt.report(next(report_dataset))


    if (logdir / "variables.pkl").exists():
        agnt.load(logdir / "variables.pkl")

    print("Train a video prediction model.")
    # for step in tqdm(range(step, config.steps), total=config.steps, initial=step):
    for _ in range(int(step.value), int(config.steps)):
        # train mae
        # print('#####################')
        # print(step.value)
        # print(config.steps)
        # print('#####################')
        mets = train_mae(next(mae_train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
        # train other part of wm
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]

        step.increment()

        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix="train")
            logger.write(fps=True)

        if should_save(step):
            agnt.save(logdir / "variables.pkl")

            agnt.wm.tssm.save(logdir / "tssm_variables.pkl", verbose=False)
            agnt.wm.wm_vit_encoder.save(logdir / "wm_vit_encoder_variables.pkl", verbose=False)
            agnt.wm.wm_vit_decoder.save(logdir / "wm_vit_decoder_variables.pkl", verbose=False)
            agnt.wm.mae_encoder.save(logdir / "mae_encoder_variables.pkl", verbose=False)
            agnt.wm.mae_decoder.save(logdir / "mae_decoder_variables.pkl", verbose=False)

    env.close()
    agnt.save(logdir / "variables.pkl")

    agnt.wm.tssm.save(logdir / "rssm_variables.pkl", verbose=False)
    agnt.wm.wm_vit_encoder.save(logdir / "wm_vit_encoder_variables.pkl", verbose=False)
    agnt.wm.wm_vit_decoder.save(logdir / "wm_vit_decoder_variables.pkl", verbose=False)
    agnt.wm.mae_encoder.save(logdir / "mae_encoder_variables.pkl", verbose=False)
    agnt.wm.mae_decoder.save(logdir / "mae_decoder_variables.pkl", verbose=False)


if __name__ == "__main__":
    main()
