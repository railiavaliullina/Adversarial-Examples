import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

from warmup_scheduler import GradualWarmupScheduler
from torch.nn import functional as f

from data.dataloader import get_dataloader
from utils.train_utils import get_optimizer, get_criterion, make_training_step
from utils.eval_utils import evaluate
from utils.debug_utils import save_batch_images, overfit_on_batch, visualize_embeddings
from utils.log_utils import start_logging, end_logging, log_metrics, log_params
from models.resnet_model import get_model
from configs.config_main import cfg as cfg_main
from configs.config_baseline import cfg as cfg_baseline
from configs.config_improved_model import cfg as cfg_improved


def train(cfg, train_dl, valid_dl, model, opt, criterion):
    """
    Trains model and logs training params
    :param cfg: config with all parameters needed for training (baseline config or improved model config)
    :param train_dl: train dataloader
    :param valid_dl: test dataloader
    :param model: ResNet-50 model (baseline or improved)
    :param opt: SGD optimizer
    :param criterion: Cross-Entropy Loss (with or without label smoothing)
    """

    # check data before training
    if cfg['debug']['save_batch']['enable']:
        save_batch_images(cfg['debug']['save_batch'], train_dl, valid_dl)

    # check training procedure before training
    if cfg['debug']['overfit_on_batch']['enable']:
        overfit_on_batch(cfg['debug']['overfit_on_batch'], cfg['train'], train_dl, model, opt, criterion)

    # save experiment name and experiment params to mlflow
    start_logging(cfg['logging'], experiment_name='baseline')
    log_params(cfg['logging'])

    # restore model if necessary
    global_step, start_epoch = 0, 0
    if cfg['logging']['load_model']:
        print(f'Trying to load checkpoint from epoch {cfg["logging"]["epoch_to_load"]}...')
        checkpoint = torch.load(cfg['logging']['checkpoints_dir'] + f'checkpoint_{cfg["logging"]["epoch_to_load"]}.pth')
        load_state_dict = checkpoint['model']
        model.load_state_dict(load_state_dict)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
        print(f'Successfully loaded checkpoint from epoch {cfg["logging"]["epoch_to_load"]}.')

    # save embeddings if enabled
    if cfg['debug']['save_embeddings']['enable']:
        visualize_embeddings(cfg['debug']['save_embeddings'], model, valid_dl, 'valid', start_epoch - 1, to_quit=True)

    # evaluate on train and test data before training
    if cfg['eval']['evaluate_before_training']:
        model.eval()
        with torch.no_grad():
            if cfg['eval']['evaluate_on_train_data']:
                evaluate(cfg['train'], cfg['logging'], model, train_dl, -1, 'train', criterion)
            evaluate(cfg['train'], cfg['logging'], model, valid_dl, -1, 'valid', criterion)
        model.train()

    # define lr_scheduler and scheduler_warmup when using improved model
    if MODEL_TYPE == 'IMPROVED':
        num_steps = cfg['train']['epochs'] * len(train_dl.dataset) // cfg['data']['dataloader']['batch_size']['train']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                                  T_max=num_steps,
                                                                  eta_min=cfg['train']['lr_scheduler']['eta_min'])
        scheduler_warmup = GradualWarmupScheduler(opt,
                                                  multiplier=cfg['train']['lr_scheduler']['multiplier'],
                                                  total_epoch=cfg['train']['lr_scheduler']['warmup_iters'],
                                                  after_scheduler=lr_scheduler)
    else:
        lr_scheduler, scheduler_warmup = None, None

    # training loop
    nb_iters_per_epoch = len(train_dl.dataset) // train_dl.batch_size

    for epoch in range(start_epoch, cfg['train']['epochs']):
        losses, reg_losses, cross_entropy_losses = [], [], []
        epoch_start_time = time.time()
        print(f'Epoch: {epoch}')
        print(f'lr at epoch {epoch}: {opt.param_groups[0]["lr"]}')
        for iter_, batch in enumerate(train_dl):
            # update scheduler_warmup and lr_scheduler
            if MODEL_TYPE == 'IMPROVED':
                lr_scheduler.step(global_step)
                scheduler_warmup.step(global_step)
                log_metrics(['train/lr'], [opt.param_groups[0]['lr']], epoch, cfg['logging'])

            loss, reg_loss, cross_entropy_loss = make_training_step(cfg['train'], batch, model, criterion, opt)
            losses.append(loss)
            reg_losses.append(reg_loss)
            cross_entropy_losses.append(cross_entropy_loss)
            global_step += 1

            log_metrics(['train/loss', 'train/reg_loss', 'train/cross_entropy_loss'],
                        [loss, reg_loss, cross_entropy_loss],
                        global_step, cfg['logging'])

            if global_step % 100 == 0:
                mean_loss = np.mean(losses[:-20]) if len(losses) > 20 else np.mean(losses)
                mean_reg_loss = np.mean(reg_losses[:-20]) if len(reg_losses) > 20 else np.mean(reg_losses)
                mean_cross_entropy_loss = np.mean(cross_entropy_losses[:-20]) if len(cross_entropy_losses) > 20 \
                    else np.mean(cross_entropy_losses)
                print(f'step: {global_step}, total_loss: {mean_loss}, cross_entropy_loss: {mean_cross_entropy_loss}, '
                      f'reg_loss: {mean_reg_loss}')

        # log mean loss per epoch
        log_metrics(['train/mean_loss'], [np.mean(losses[:-nb_iters_per_epoch])], epoch, cfg['logging'])
        print(f'Epoch time: {round((time.time() - epoch_start_time) / 60, 3)} min')

        # save model
        if cfg['logging']['save_model'] and epoch % cfg['logging']['save_frequency'] == 0:
            print('Saving current model...')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'opt': opt.state_dict(),
            }
            torch.save(state, cfg['logging']['checkpoints_dir'] + f'checkpoint_{epoch}.pth')

        # evaluate on train and test data
        model.eval()
        with torch.no_grad():
            if cfg['eval']['evaluate_on_train_data']:
                evaluate(cfg['train'], cfg['logging'], model, train_dl, epoch, 'train', criterion)
            evaluate(cfg['train'], cfg['logging'], model, valid_dl, epoch, 'valid', criterion)
        model.train()

    end_logging(cfg['logging'])


def run(cfg):
    """
    Sets up parameters for training and runs training
    :param cfg: config with all parameters needed for training (baseline config or improved model config)
    """
    # train_dl = get_dataloader(cfg['data'], 'train')
    valid_dl = get_dataloader(cfg['data'], 'valid')

    model = get_model(cfg['model'], MODEL_TYPE)

    # restore model if necessary
    global_step, start_epoch = 0, 0
    if cfg['logging']['load_model']:
        print(f'Trying to load checkpoint from epoch {cfg["logging"]["epoch_to_load"]}...')
        checkpoint = torch.load(cfg['logging']['checkpoints_dir'] + f'checkpoint_{cfg["logging"]["epoch_to_load"]}.pth')
        load_state_dict = checkpoint['model']
        model.load_state_dict(load_state_dict)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
        print(f'Successfully loaded checkpoint from epoch {cfg["logging"]["epoch_to_load"]}.')

    idx_to_class = {v: k for k, v in valid_dl.dataset.class_to_idx.items()}
    epsilons = [0.01]  # [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0001, 0.0005]
    for epsilon in epsilons[::-1]:
        correct = 0
        adv_examples = []
        adv_examples_ = []
        adv_ex_i = 0
        for i, (data, target) in enumerate(valid_dl):
            target = torch.tensor([int(target[0])])
            print(i)
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            loss = f.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            sign_data_grad = data_grad.sign()
            perturbed_data = data + epsilon * sign_data_grad
            # perturbed_data = torch.clamp(perturbed_image, 0, 1)
            output = f.softmax(model(perturbed_data))

            final_pred = output.max(1, keepdim=True)[1].squeeze(-1)  # get the index of the max log-probability
            max_prob = output.squeeze(0)[output.argmax(1)]

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()

            if final_pred.item() == target.item():
                correct += 1
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_examples.append(
                        (idx_to_class[init_pred.item()], idx_to_class[final_pred.item()], max_prob, adv_ex))
            else:
                source = torch.clamp(data, 0, 1).squeeze().detach().cpu().numpy()
                adv_examples.append((idx_to_class[init_pred.item()], idx_to_class[final_pred.item()], max_prob, adv_ex))

                if max_prob.item() > 0.7:
                    # plt.imsave(f'../plots/'
                    #            f'eps_{epsilon}_{adv_ex_i}_{idx_to_class[target.item()]}'
                    #            f'_{idx_to_class[init_pred.item()]}'
                    #            f'_{idx_to_class[final_pred.item()]}'
                    #            f'_{max_prob.item()}.png', adv_ex.transpose((1, 2, 0)))
                    # plt.imsave(f'../source/'
                    #            f'{idx_to_class[target.item()]}.png', source.transpose((1, 2, 0)))
                    adv_ex_i += 1

            # plt.imsave(f'../plots1/'
            #            f'eps_{epsilon}_{adv_ex_i}_{idx_to_class[target.item()]}'
            #            f'_{idx_to_class[init_pred.item()]}'
            #            f'_{idx_to_class[final_pred.item()]}'
            #            f'_{max_prob.item()}.png', adv_ex.transpose((1, 2, 0)))
            adv_examples_.append(adv_ex)
        final_acc = correct / float(len(valid_dl))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(valid_dl), final_acc))

        # with open('../plots1/adv_examples.pickle', 'wb') as f:
        #     pickle.dump(adv_examples_, f)

    a = 1

    # opt = get_optimizer(cfg, model, MODEL_TYPE)
    # criterion = get_criterion(cfg, MODEL_TYPE)
    # run training
    # train(cfg, train_dl, valid_dl, model, opt, criterion)


if __name__ == '__main__':
    MODEL_TYPE = cfg_main['model_type'].name
    print(f'MODEL TYPE: {MODEL_TYPE}')

    # define config based on using model (baseline or improved)
    cfg = cfg_baseline if MODEL_TYPE == 'BASELINE' else cfg_improved

    # run training
    start_time = time.time()
    run(cfg)
    print(f'Total time: {round((time.time() - start_time) / 60, 3)} min')
