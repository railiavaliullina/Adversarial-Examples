import time
import numpy as np
import torch

from utils.log_utils import log_metrics


def evaluate(cfg_train, cfg_logging, model, dl, epoch, dataset_type, criterion):
    """
    Evaluates on train/valid data
    :param cfg_eval: cfg['train'] part of config
    :param cfg_logging: cfg['logging'] part of config
    :param model: resnet-50 model
    :param dl: train/valid dataloader
    :param epoch: epoch for logging
    :param dataset_type: type of current data ('train' or 'valid')
    """
    print(f'Evaluating on {dataset_type} data...')
    eval_start_time = time.time()
    correct, total = 0, 0
    cross_entropy_losses, reg_losses, losses = [], [], []
    unique_labels = np.unique(dl.dataset.labels)
    accuracies_for_classes = [0 for _ in unique_labels]
    model = model.cuda()

    dl_len = len(dl)
    for i, (images, labels) in enumerate(dl):
        images, labels = images.cuda(), labels.cuda()

        if i % 50 == 0:
            print(f'iter: {i}/{dl_len}')

        logits = model(images)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += torch.sum(predicted == labels)

        for i, l in enumerate(labels):
            accuracies_for_classes[l] += torch.sum((predicted[i] == l))

        # calculate losses
        cross_entropy_loss = criterion(logits, labels)
        cross_entropy_losses.append(cross_entropy_loss.item())
        l2_reg = torch.tensor(0.0, requires_grad=False)
        for name, param in model.named_parameters():
            if '.bias' not in name and '.bn' not in name:  # no biases or BN params
                l2_reg = l2_reg + param.norm(2)
        l2_reg *= cfg_train['opt']['weight_decay']
        reg_losses.append(l2_reg.item())
        losses.append((cross_entropy_loss + l2_reg).item())

    log_metrics([f'{dataset_type}_eval/cross_entropy_loss', f'{dataset_type}_eval/reg_loss_train',
                 f'{dataset_type}_eval/total_loss_train'],
                [np.mean(cross_entropy_losses), np.mean(reg_losses), np.mean(losses)], epoch, cfg_logging)

    accuracy = 100 * correct.item() / total
    print(f'Accuracy on {dataset_type} data: {accuracy}')
    accuracies_for_classes = [100 * acc.item() / dl.dataset.nb_images_per_class[i] for i, acc in
                              enumerate(accuracies_for_classes)]
    print(f'accuracies for classes: {accuracies_for_classes}')
    balanced_acc = sum(accuracies_for_classes) / dl.dataset.nb_classes
    print(f'Balanced accuracy: {balanced_acc}')

    for i, acc in enumerate(accuracies_for_classes):
        log_metrics([f'{dataset_type}_eval/accuracy_class_{i}'], [acc], epoch, cfg_logging)

    log_metrics([f'{dataset_type}_eval/accuracy', f'{dataset_type}_eval/balanced_accuracy'], [accuracy, balanced_acc],
                epoch, cfg_logging)
    print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')
