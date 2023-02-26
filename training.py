import os
from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load

from dataset import DatasetMimic
from neural_network import NeuralNetwork, NeuralNetworkWithSoftmax
from utils.constants import CHECKPOINTS_DIRECTORY

DEFAULT_CONFIG = "config_training.yml"

# TODO: Solve the error in the policy

def string_batch_information(batch_index, batches, total, total_loss, correct_policy, correct_value, time_loss):
    """
    Genera una cadena con la información del batch
    :param batch_index: Índice del batch
    :param batches: Número de batches totales
    :param total: Número de posiciones analizadas hasta el momento
    :param total_loss: Pérdida total hasta el momento
    :param correct_policy: Número de movimientos correctos predecidos hasta el momento
    :param correct_value: Número de resultados correctos predecidos hasta el momento
    :param time_loss: Pérdida del tiempo hasta el momento
    :return: Cadena con la información del batch
    """
    return f"Batch {batch_index + 1}/{batches}: \n" \
           f"Pérdida total: {total_loss / (batch_index + 1)} \n " \
           f"Precisión de la política: {100. * correct_policy / total:.2f}% \n " \
           f"Precisión del valor: {100. * correct_value / total:.2f} \n" \
           f"Pérdida del tiempo: {time_loss / (batch_index + 1)} \n{100 * '-'}"


def load_neural_network(name: str, with_softmax=False) -> Tuple[NeuralNetwork, Optional[Dict[str, Any]]]:
    """
    Loads the weights of the neural network
    :param name: Name of the checkpoint
    :param with_softmax: Indicates if the neural network with softmax is loaded
    :return: Neural network and dictionary with the weights
    """

    net = NeuralNetwork()

    path_checkpoint = f"{CHECKPOINTS_DIRECTORY}/{name}.pth"

    if not os.path.exists(path_checkpoint):
        print('There are no weights of the neural network stored from supervised learning')

        if not os.path.isdir(CHECKPOINTS_DIRECTORY):
            os.mkdir(CHECKPOINTS_DIRECTORY)

        state = None

    else:
        map_location = 'cpu' if not torch.cuda.is_available() else None

        print('Loading weights of the neural network from supervised learning')
        state = torch.load(path_checkpoint, map_location=map_location)
        net.load_state_dict(state['net'])

    if with_softmax:
        net = NeuralNetworkWithSoftmax(net)

    return net, state


def save_neural_network(name: str, epoch: int, net, optimizer, scheduler, best_accu: Optional[float], use_gpu):
    """
    Saves the weights of the neural network
    :param name: Name of the checkpoint
    :param epoch: Epoch
    :param net: Neural network
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param best_accu: Best accuracy
    :param use_gpu: Indicates if the neural network is trained in GPU
    """
    state = {
        'net': net.state_dict() if not use_gpu else net.module.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_accuracy': best_accu,
    }
    path_checkpoint = f"{CHECKPOINTS_DIRECTORY}/{name}.pth"
    torch.save(state, path_checkpoint)


def train_epoch(net: Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion_policy: nn.CrossEntropyLoss,
                criterion_value: nn.CrossEntropyLoss,
                criterion_time: nn.MSELoss,
                device: torch.device) -> Tuple[float, float, float, float]:
    """
    Entrena la red neuronal usando los parámetros dados

    :param net: red neuronal a entrenar
    :param dataloader: dataloader con el dataset
    :param optimizer: optimizador
    :param criterion_policy: función de pérdida para la política
    :param criterion_value: función de pérdida para el valor
    :param criterion_time: función de pérdida para el tiempo
    :param device: dispositivo en el que ejecutar el entrenamiento
    :return: tupla con la pérdida total, la precisión de la política, precisión del valor y la pérdida del tiempo
    """
    net.train()

    total_train_loss = 0

    correct_policy, correct_value, loss_time = 0, 0, 0

    total = 0

    for batch_index, (inputs, targets) in enumerate(dataloader):
        position, action_mask, times = inputs
        target_policy, target_value, target_time = targets

        position = position.to(device)
        action_mask = action_mask.to(device)
        times = times.to(device)

        target_policy = target_policy.to(device)
        target_value = target_value.to(device)
        target_time = target_time.to(device)

        optimizer.zero_grad()
        policy, value, time = net((position, action_mask, times))
        time = torch.squeeze(time)

        loss_policy = criterion_policy(policy, target_policy)
        loss_value = criterion_value(value, target_value)
        loss_time = criterion_time(time, target_time)
        loss = loss_policy + loss_value + loss_time

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total += target_policy.size(0)
        correct_policy += target_policy.eq(policy.max(1)[1]).sum().item()
        correct_value += target_value.eq(value.max(1)[1]).sum().item()
        loss_time += loss_time.item()

        print(string_batch_information(
            batch_index,
            len(dataloader),
            total,
            total_train_loss,
            correct_policy,
            correct_value,
            loss_time
        ))

    return total_train_loss / len(dataloader), 100. * correct_policy / total, \
        100. * correct_value / total, loss_time / len(dataloader)


def validation_epoch(net: Module,
                     dataloader: DataLoader,
                     criterion_policy: nn.CrossEntropyLoss,
                     criterion_value: nn.CrossEntropyLoss,
                     criterion_time: nn.MSELoss,
                     device: torch.device) -> Tuple[float, float, float, float]:
    """
    Validación de un epoch de la red neuronal

    :param net: red neuronal a evaluar
    :param dataloader: dataloader con el dataset de validación
    :param criterion_policy: función de pérdida para la política
    :param criterion_value: función de pérdida para el valor
    :param criterion_time: función de pérdida para el tiempo
    :param device: dispositivo en el que ejecutar la validación
    :return: tupla con la pérdida total, la precisión de la política y pérdida del valor
    """
    net.eval()

    total_val_loss = 0

    correct_policy, correct_value, loss_time = 0, 0, 0

    total = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            position, action_mask, times = inputs
            target_policy, target_value, target_time = targets

            position = position.to(device)
            action_mask = action_mask.to(device)
            times = times.to(device)

            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            target_time = target_time.to(device)

            policy, value, time = net((position, action_mask, times))
            time = torch.squeeze(time)

            loss_policy = criterion_policy(policy, target_policy)
            loss_value = criterion_value(value, target_value)
            loss_time = criterion_time(time, target_time)
            loss = loss_policy + loss_value + loss_time

            total_val_loss += loss.item()
            total += target_policy.size(0)
            correct_policy += target_policy.eq(policy.max(1)[1]).sum().item()
            correct_value += target_value.eq(value.max(1)[1]).sum().item()
            loss_time += loss_time.item()

            print(string_batch_information(
                batch_index,
                len(dataloader),
                total,
                total_val_loss,
                correct_policy,
                correct_value,
                loss_time
            ))

    return total_val_loss / len(dataloader), 100. * correct_policy / total, \
        100. * correct_value / total, loss_time / len(dataloader)


def train(config: Dict[str, Any], name_training: str):
    """
    Training of the neural network
    :param config: dictionary with the configuration for the training
    :param name_training: name of the training
    """
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        cudnn.benchmark = True

    device = torch.device("cuda" if gpu_available else "cpu")

    print("Loading dataset... (can take a while)")
    dataset_training = DatasetMimic(config['dataset_training_folder'])
    dataset_validation = DatasetMimic(config['dataset_validation_folder'])

    net, state = load_neural_network(name_training, with_softmax=False)
    net.to(device)

    if gpu_available:
        net = torch.nn.DataParallel(net)

    dataloader_training = DataLoader(dataset_training, batch_size=config['batch'], shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=config['batch'], shuffle=True)

    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.CrossEntropyLoss()
    criterion_time = torch.nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.0001)
    milestones = [int(config['epochs'] * 0.2), int(config['epochs'] * 0.5), int(config['epochs'] * 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if state:
        start_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_accu = state['best_accuracy']

    else:
        start_epoch = 1
        best_accu = 0

        print(f'A model could not be loaded with name {name_training}. Starting training from scratch')

    writer = SummaryWriter(log_dir=f"runs/{name_training}")

    print(f"Training {name_training} with the following parameters:")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch']}")
    print(f"Learning rate: {config['lr']}")

    for epoch in range(start_epoch, config['epochs'] + 1):

        print(f"Epoch {epoch}/{config['epochs']}")
        print("Starting training...")

        total_loss, policy_accu, value_accu, time_loss = train_epoch(
            net,
            dataloader_training,
            optimizer,
            criterion_policy,
            criterion_value,
            criterion_time,
            device
        )

        print("Training finished. Starting validation...")

        total_loss_val, policy_accu_val, value_loss_val, time_loss_val = validation_epoch(
            net,
            dataloader_validation,
            criterion_policy,
            criterion_value,
            criterion_time,
            device
        )

        writer.add_scalar("Total loss/training", total_loss, epoch)
        writer.add_scalar("Accuracy of policy/training ", policy_accu, epoch)
        writer.add_scalar("Accuracy of value/training", value_accu, epoch)
        writer.add_scalar("Time loss/training", time_loss, epoch)

        writer.add_scalar("Total loss/validation", total_loss_val, epoch)
        writer.add_scalar("Accuracy of policy/validation", policy_accu_val, epoch)
        writer.add_scalar("Accuracy of value/validation", value_loss_val, epoch)
        writer.add_scalar("Time loss/validation", time_loss_val, epoch)

        writer.flush()

        print("Validation finished. Saving model...")

        if policy_accu_val > best_accu:
            print("The model has improved. Saving model...")
            print(f"Best accuracy: {policy_accu_val}")
            save_neural_network(name_training, epoch, net, optimizer, scheduler, policy_accu_val, gpu_available)
            best_accu = policy_accu_val

        scheduler.step()

    writer.close()


def main():
    """Training of the neural network using supervised learning"""
    import argparse

    parser = argparse.ArgumentParser(description='Script for training the neural network')

    parser.add_argument('name_training', type=str, help='Name of the training (used for the checkpoints)')

    parser.add_argument(
        '--config-filename',
        default=DEFAULT_CONFIG,
        type=str,
        help='Config file for the training'
    )

    args = parser.parse_args()
    config = safe_load(open(args.config_filename, 'r'))

    train(config, name_training=args.name_training)


if __name__ == '__main__':
    main()
