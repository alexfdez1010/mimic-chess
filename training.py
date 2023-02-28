from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn, optim, Tensor
from torch.backends import cudnn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load

from dataset import DatasetMimic
from neural_network import load_neural_network
from utils.constants import CHECKPOINTS_DIRECTORY

DEFAULT_CONFIG = "config_training.yml"


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
           f"Precisión del valor: {100. * correct_value / total:.2f}% \n" \
           f"Pérdida del tiempo: {time_loss / (batch_index + 1)} \n{100 * '-'}"


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

        position, action_mask, times = prepare_input(device, inputs)
        target_policy, target_value, target_time = prepare_output(device, targets)

        optimizer.zero_grad()
        policy, value, time = net((position, action_mask, times))

        loss, loss_time = compute_loss(
            criterion_policy, criterion_value, criterion_time,
            target_policy, target_value, target_time,
            policy, value, time
        )

        loss.backward()
        optimizer.step()

        total, total_train_loss, correct_policy, correct_value, loss_time = update_metrics(
            total, total_train_loss,
            correct_policy, correct_value,
            loss, loss_time,
            target_policy, policy,
            target_value, value
        )

        print(string_batch_information(
            batch_index,
            len(dataloader),
            total,
            total_train_loss,
            correct_policy,
            correct_value,
            loss_time
        ))

    mean_loss = total_train_loss / len(dataloader)
    mean_accuracy_policy = 100. * correct_policy / total
    mean_accuracy_value = 100. * correct_value / total

    return mean_loss, mean_accuracy_policy, mean_accuracy_value, loss_time


def update_metrics(total: int,
                   total_train_loss: float,
                   correct_policy: int,
                   correct_value: int,
                   loss: Tensor,
                   loss_time: Tensor,
                   target_policy: Tensor,
                   policy: Tensor,
                   target_value: Tensor,
                   value: Tensor):
    """
    Updates the metrics
    :param total: total number of positions analyzed
    :param total_train_loss: total loss
    :param correct_policy: number of moves predicted correctly
    :param correct_value: number of results predicted correctly
    :param loss: loss obtained in the batch
    :param loss_time: loss of the time obtained in the batch
    :param target_policy: moves target in the batch
    :param policy: moves predicted in the batch
    :param target_value: results target in the batch
    :param value: results predicted in the batch
    :return: updated metrics (total, total_train_loss, correct_policy, correct_value, loss_time)
    """
    total_train_loss += loss.item()
    total += target_policy.size(0)

    correct_policy += target_policy.eq(policy.max(1)[1]).sum().item()
    correct_value += target_value.eq(value.max(1)[1]).sum().item()
    loss_time += loss_time.item()

    return total, total_train_loss, correct_policy, correct_value, loss_time


def compute_loss(
        criterion_policy: nn.CrossEntropyLoss,
        criterion_value: nn.CrossEntropyLoss,
        criterion_time: nn.MSELoss,
        target_policy: Tensor,
        target_value: Tensor,
        target_time: Tensor,
        policy: Tensor,
        value: Tensor,
        time: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the loss of the neural network
    :param criterion_policy: loss function for the policy
    :param criterion_value: loss function for the value
    :param criterion_time: loss function for the time
    :param target_policy: policy target
    :param target_value: value target
    :param target_time: time target
    :param policy: policy predicted by the neural network
    :param value: value predicted by the neural network
    :param time: time predicted by the neural network
    :return: a tuple with the total loss and the time loss
    """

    time = torch.squeeze(time)

    loss_policy = criterion_policy(policy, target_policy)
    loss_value = criterion_value(value, target_value)
    loss_time = criterion_time(time, target_time)
    loss = loss_policy + loss_value + loss_time

    return loss, loss_time


def prepare_input(device, inputs):
    """
    Prepare the input and output for the neural network
    :param device: device where the neural network is trained
    :param inputs: input of the neural network
    :return: input of the neural network
    """
    position, action_mask, times = inputs

    position = position.to(device)
    action_mask = action_mask.to(device)
    times = times.to(device)

    return position, action_mask, times


def prepare_output(device, targets):
    """
    Prepare the input and output for the neural network
    :param device: device where the neural network is trained
    :param targets: output of the neural network
    :return: output of the neural network
    """
    target_policy, target_value, target_time = targets

    target_policy = target_policy.to(device)
    target_value = target_value.to(device)
    target_time = target_time.to(device)

    return target_policy, target_value, target_time


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
            position, action_mask, times = prepare_input(device, inputs)
            target_policy, target_value, target_time = prepare_output(device, targets)

            policy, value, time = net((position, action_mask, times))

            loss, loss_time = compute_loss(
                criterion_policy, criterion_value, criterion_time,
                target_policy, target_value, target_time,
                policy, value, time
            )

            total, total_val_loss, correct_policy, correct_value, loss_time = update_metrics(
                total, total_val_loss,
                correct_policy, correct_value,
                loss, loss_time,
                target_policy, policy,
                target_value, value
            )

            print(string_batch_information(
                batch_index,
                len(dataloader),
                total,
                total_val_loss,
                correct_policy,
                correct_value,
                loss_time
            ))

    mean_loss = total_val_loss / len(dataloader)
    mean_accuracy_policy = 100. * correct_policy / total
    mean_accuracy_value = 100. * correct_value / total

    return mean_loss, mean_accuracy_policy, mean_accuracy_value, loss_time


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

        print("Validation finished")

        if policy_accu_val > best_accu:
            print("The model has improved. Saving model...")
            print(f"The accuracy has improved from {best_accu:.2f} to {policy_accu_val:.2f}")
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
