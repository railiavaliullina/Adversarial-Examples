import mlflow


def start_logging(cfg, experiment_name=None):
    """
    Starts mlflow logging
    :param cfg: cfg['logging'] part of config
    :param experiment_name: experiment name for mlflow visualization
    """
    if cfg['log_metrics']:
        experiment_name = cfg['train']['experiment_name'] if experiment_name is None else experiment_name
        mlflow.start_run(run_name=experiment_name)


def end_logging(cfg):
    """
    Finishes mlflow logging
    :param cfg: cfg['logging'] part of config
    """
    if cfg['log_metrics']:
        mlflow.end_run()


def log_metrics(names, metrics, step, cfg):
    """
    Logs metrics in given list with corresponding names
    :param names: list of names of given metrics
    :param metrics: list of given metrics
    :param step: step to log
    :param cfg: cfg['logging'] part of config
    """
    if cfg['log_metrics']:
        for name, metric in zip(names, metrics):
            mlflow.log_metric(name, metric, step)


def log_params(cfg):
    """
    Logs experiment config with all parameters
    :param cfg: cfg['logging'] part of config
    """
    if cfg['log_metrics']:
        mlflow.log_param('cfg', cfg)
