import torch
from losses.LabelSmoothLoss import LabelSmoothLoss


def get_optimizer(cfg, model, model_type):
    """
    Gets optimizer for parameters update
    :param cfg: config with all parameters needed for training (baseline config or improved model config)
    :param model: ResNet-50 model
    :param model_type: type of using model (baseline or improved)
    :return: optimizer
    """
    if cfg['train']['opt']['optim_type'] == 'SGD':
        init_lr = cfg['train']['opt']['learning_rate']
        lr = init_lr if model_type == 'BASELINE' \
            else init_lr * cfg['data']['dataloader']['batch_size']['train'] / cfg['train']['opt']['resnet_batch_size']
        opt = torch.optim.SGD(params=model.parameters(),
                              lr=lr,
                              momentum=cfg['train']['opt']['momentum'],
                              # weight_decay=cfg['train']['opt']['weight_decay'],
                              nesterov=cfg['train']['opt']['nesterov'])
    else:
        raise Exception
    return opt


def get_criterion(cfg, model_type):
    """
    Gets loss function
    :param cfg: config with all parameters needed for training (baseline config or improved model config)
    :param model_type: type of using model (baseline or improved)
    :return: loss function
    """
    criterion = LabelSmoothLoss(smooth_eps=cfg['train']['loss']['smooth_eps']) if model_type == 'IMPROVED' \
        else torch.nn.CrossEntropyLoss()
    return criterion


def make_training_step(cfg_train, batch, model, criterion, optimizer):
    """
    Makes single parameters updating step.
    :param cfg_train: cfg['train'] part of config
    :param batch: current batch
    :param model: resnet50 model
    :param criterion: criterion
    :param optimizer: optimizer
    :param iter_: current iteration
    :return: current loss value
    """
    images, labels = batch
    images, labels, model = images.cuda(), labels.cuda(), model.cuda()
    optimizer.zero_grad()
    logits = model(images)
    cross_entropy_loss = criterion(logits, labels)
    l2_reg = torch.tensor(0.0, requires_grad=False)
    for name, param in model.named_parameters():
        if '.bias' not in name and '.bn' not in name:  # no biases or BN params
            l2_reg = l2_reg + param.norm(2)
    l2_reg *= cfg_train['opt']['weight_decay']
    loss = cross_entropy_loss + l2_reg
    loss.backward()
    optimizer.step()
    return loss.item(), l2_reg.item(), cross_entropy_loss.item()
