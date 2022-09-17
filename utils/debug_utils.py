import torchvision
import torch
import numpy as np
from tensorboardX import SummaryWriter
import math


def overfit_on_batch(cfg_overfit_on_batch, cfg_train, train_dl, model, optimizer, criterion):
    """
    Overfits on one batch
    :param cfg_overfit_on_batch: cfg['debug']['overfit_on_batch'] part of config
    :param train_dl: train dataloader
    :param model: resnet50 model
    :param optimizer: optimizer
    :param criterion: criterion
    """
    train_dl = iter(train_dl)
    images, labels = next(train_dl)
    model = model.cuda()
    accuracies = []

    for iter_ in range(cfg_overfit_on_batch['nb_iters']):
        optimizer.zero_grad()
        logits = model(images.cuda()).cpu()
        # calculate loss
        cross_entropy_loss = criterion(logits, labels)
        l2_reg = torch.tensor(0.0, requires_grad=False)
        for name, param in model.named_parameters():
            if '.bias' not in name and '.bn' not in name:  # no biases or BN params
                l2_reg = l2_reg + param.norm(2)
        l2_reg *= cfg_train['opt']['weight_decay']
        loss = cross_entropy_loss + l2_reg
        # calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        accuracy = torch.sum(predicted == labels).item() / labels.size(0) * 100
        print(
            f'iter: {iter_}, acc: {accuracy}, cross_entropy_loss: {cross_entropy_loss.item()}, l2_reg: {l2_reg.item()}, '
            f'total loss: {loss.item()}')

        accuracies.append(accuracy)
        if len(accuracies) >= 5 and np.min(accuracies[-5:]) == 100:
            break

        loss.backward()
        optimizer.step()
    print(f'Overfitting on batch is finished.')


def save_batch_images(cfg, train_dl, valid_dl):
    """
    Saves several batches of images as .png file
    :param cfg: cfg['debug']['save_batch'] part of config
    :param train_dl: train dataloader to saves batches from
    :param valid_dl: valid dataloader to saves batches from
    """
    for dl in [train_dl, valid_dl]:
        dataset_type = dl.dataset.dataset_type
        print(dataset_type)
        dl = iter(dl)
        for i in range(cfg['nrof_batches_to_save']):
            images, labels = next(dl)
            print(f'batch {i} labels: {labels}')
            torchvision.utils.save_image(images, cfg['path_to_save'] + f'{dataset_type}_batch_{i}.png')


def visualize_embeddings(cfg, model, dataloader, dataset_type, epoch, to_quit=False):
    embeddings_writer = SummaryWriter(log_dir=cfg['tensorboard_dir'] + f'/embeddings_vis/epoch_{epoch}')
    num_images = cfg['num_images']
    tag = f'{dataset_type}_batch_after_epoch_{epoch}'
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, shuffle=True)

    all_images, all_embeddings, all_labels = [], [], []
    dl = iter(dataloader)
    for i in range(num_images):
        batch = next(dl)
        images, labels = batch[0], batch[1]
        embeddings = model(images)
        all_images.extend(images)
        all_embeddings.extend(embeddings)
        all_labels.extend(labels)

    embeddings_writer.add_embedding(np.asarray([e.detach().numpy() for e in all_embeddings]),
                                    metadata=np.asarray([a.item() for a in all_labels]), tag=tag,
                                    label_img=np.asarray([im.numpy() for im in all_images]))
    print(f'Successfully saved embeddings visualization!')
    if to_quit:
        quit()


def get_layer_receptive_field(params, layer):
    n_in, j_in, r_in = layer
    k, s, p = params
    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    return (n_out, j_out, r_out)


def get_receptive_field():
    # k_size, stride, padding
    input_size = 224
    conv_params = [(7, 2, 3), (3, 2, 1), (1, 1, 0), (3, 1, 1), (1, 1, 0), (1, 1, 0),
                   (1, 1, 0), (3, 1, 1), (1, 1, 0)]
    layers = ['conv1', 'pool1', 'layer1_conv1', 'layer1_conv2', 'layer1_conv3', 'layer1_downsample_conv',
              'conv1', 'conv2', 'conv3']

    layer = (input_size, 1, 1)
    for i in range(len(layers)):
        layer = get_layer_receptive_field(conv_params[i], layer)
        r = layer[2]
        print(f'layer: {layers[i]}, receptive field: {(r, r)}')


# if __name__ == '__main__':
#     get_receptive_field()
