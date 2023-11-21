import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Union

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from attack.attack_pgd import PGD_triplet

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from typing import Tuple, Dict
# from src.utils import get_embeddings_from_dataloader

EPS = 30 / 255
ALPHA = 2 / 225
STEPS = 20

###
@torch.no_grad()
### USE IN 'lenet_model' 'new_val'
def get_embeddings_from_dataloader(loader: DataLoader, model: nn.Module, device: torch.device,
                                   return_numpy_array=False, return_image_paths=False,
                                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)
        embeddings: torch.Tensor = model(images)
        embeddings_ls.append(embeddings)
        labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings, labels, images_paths)

    return (embeddings, labels)


### USE IN 'lenet_model' 'new_val_pgd'
def get_embeddings_from_images(images_, labels_, model: nn.Module,
                               device: torch.device, return_numpy_array=False, return_image_paths=False,
                               ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    # for images_, labels_ in loader:
    images: torch.Tensor = images_.to(device, non_blocking=True)
    labels: torch.Tensor = labels_.to(device, non_blocking=True)
    embeddings: torch.Tensor = model(images)
    embeddings_ls.append(embeddings)
    labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.cpu().numpy()

    # if return_image_paths:
    #     images_paths: List[str] = []
    #     for path, _ in loader.dataset.samples:
    #         images_paths.append(path)
    #     return (embeddings, labels, images_paths)

    return (embeddings, labels)

def TSNE_plot_embedding(embeddings, labels, kind):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # colors = ['#a1d6d6', '#d4bfdd', '#a0a6cc', '#f5d9bd', '#9dcfb6',
    #           '#90abdb', '#dbd2b1', '#e0b7bd', '#c1b4d3', '#ccac9b']
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    for i in range(10):
        inds = np.where(labels == i)[0]
        #plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
        #plt.scatter(X_tsne[inds, 0], X_tsne[inds, 1], c=labels, label=i, color=colors[i])
        plt.scatter(X_tsne[inds, 0], X_tsne[inds, 1], label=i, color=colors[i], alpha=0.8)
    plt.legend()
    plt.title(kind)
    plt.savefig('plot/' + kind + '_tsne.png')
    print('>>> tSNE save to plot/' + kind + '_tsne.png')
    plt.show()

def imshow(images, labels, kind, print_label=True):
    # dataiter = iter(loader)
    # images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    if (torch.cuda.is_available() is False):
        np_img = img.numpy()
    else:
        np_img = img.cpu().numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(kind)
    plt.savefig('plot/' + kind + '_data.png')
    # if kind=='CLEAN':
    #     plt.savefig('plot/clean_data.png')
    # else:
    #     plt.savefig('plot/poison_data.png')
    if print_label:
        print(labels)
    plt.show()


### USE
def knn_acc(embeddings_train, labels_train, embeddings_test, labels_test):
    correct = []
    test_acc = 0.0
    for k in range(embeddings_test.shape[0]):
        pdist = (embeddings_train - embeddings_test[k].view(1, embeddings_test.shape[1])).norm(dim=1)
        train_topk = pdist.topk(k=5, largest=False)[1]

        nearest_n = []
        for label in train_topk:
            nearest_n.append(labels_train[label].item())
        count_max = np.bincount(nearest_n)  # count=[0 0 0 0 0 0 2 2]
        num_max = np.bincount(count_max)  # 2 最大值的个数
        if num_max[max(count_max)] == 1:
            pre_label = np.argmax(count_max)
        else:
            pre_label = labels_train[train_topk[0]].item()
        correct.append((pre_label == labels_test[k]).item())
    correct = np.sum(correct)
    total = len(labels_test)
    test_acc = 100. * correct / total
    return test_acc

### USE
def knn_pred(embeddings_train, labels_train, embeddings_test):
    predict = []
    for k in range(embeddings_test.shape[0]):
        pdist = (embeddings_train - embeddings_test[k].view(1, embeddings_test.shape[1])).norm(dim=1)
        train_topk = pdist.topk(k=5, largest=False)[1]
        # print(f'len(pdist)={len(pdist)}')
        # print(f'topk={train_topk}')
        nearest_n = []
        for label in train_topk:
            nearest_n.append(labels_train[label].item())
        count_max = np.bincount(nearest_n)  # count=[0 0 0 0 0 0 2 2]
        num_max = np.bincount(count_max)  # 2 最大值的个数
        if num_max[max(count_max)] == 1:
            pre_label = np.argmax(count_max)
        else:
            pre_label = labels_train[train_topk[0]].item()
        predict.append(pre_label)
    return predict

# USE IN EXPERIMENT
def get_pgd_emb_from_dataloader(loader: DataLoader, model: nn.Module, device: torch.device,
                                return_numpy_array=False, return_image_paths=False,
                                ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    atk_test = PGD_triplet(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
    print(atk_test)

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)

        labels = torch.squeeze(labels, dim=1)
        adv_images = atk_test(images, labels)

        embeddings: torch.Tensor = model(adv_images)
        embeddings_ls.append(embeddings)
        labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings, labels, images_paths)

    return (embeddings, labels)








###############################################
####### NOT USE IN FINAL VERSION ##############

# NOT USE
def calculate_all_metrics(model: nn.Module,
                          test_loader: DataLoader,
                          ref_loader: DataLoader,
                          device: torch.device,
                          k: Tuple[int, int, int] = (1, 5, 10)
                          ) -> Dict[str, float]:

    # Calculate all embeddings of training set and test set
    model.eval()
    embeddings_test, labels_test = get_embeddings_from_dataloader(test_loader, model, device)
    embeddings_ref, labels_ref = get_embeddings_from_dataloader(ref_loader, model, device)

    # Expand dimension for batch calculating
    embeddings_test = embeddings_test.unsqueeze(dim=0)  # [M x K] -> [1 x M x embedding_size]
    #embeddings_ref = embeddings_ref.unsqueeze(dim=0)  # [N x K] -> [1 x N x embedding_size]
    labels_test = labels_test.unsqueeze(dim=1)  # [M] -> [M x 1]

    # Pairwise distance of all embeddings between test set and reference set
    distances: torch.Tensor = torch.cdist(embeddings_test, embeddings_ref, p=2).squeeze()  # [M x N]

    # Calculate precision_at_k on test set with k=1, k=5 and k=10
    metrics: Dict[str, float] = {}
    # for i in k:
    #     metrics[f"average_precision_at_{i}"] = calculate_precision_at_k(distances,
    #                                                                     labels_test,
    #                                                                     labels_ref,
    #                                                                     k=i
    #                                                                     )
    # # Calculate mean average precision (MAP)
    # mean_average_precision: float = sum(precision_at_k for precision_at_k in metrics.values()) / len(metrics)
    # metrics["mean_average_precision"] = mean_average_precision

    # Calculate top-1 and top-5 and top-10 accuracy
    for i in k:
        metrics[f"top_{i}_accuracy"] = calculate_topk_accuracy(distances,
                                                               labels_test,
                                                               labels_ref,
                                                               top_k=i
                                                               )
    # Calculate NMI score
    #n_classes: int = len(test_loader.dataset.classes)
    # metrics["normalized_mutual_information"] = calculate_normalized_mutual_information(
    #     embeddings_test.squeeze(), labels_test.squeeze(), n_classes
    # )

    return metrics

# NOT USE
def calculate_topk_accuracy(distances: torch.Tensor,
                            labels_test: torch.Tensor,
                            labels_ref: torch.Tensor,
                            top_k: int
                            ) -> float:

    _, indices = distances.topk(k=top_k, dim=1, largest=False)  # indices shape: [M x k]

    y_pred = []
    for i in range(top_k):
        indices_at_k: torch.Tensor = indices[:, i]  # [M]
        y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(dim=1)  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
    labels_test = torch.hstack((labels_test,) * top_k)  # [M x k]

    n_predictions: int = y_pred.shape[0]
    n_true_predictions: int = ((y_pred == labels_test).sum(dim=1) > 0).sum().item()
    topk_accuracy: float = n_true_predictions / n_predictions * 100
    return topk_accuracy

# def calculate_topk_accuracy(distances: torch.Tensor,
#                             labels_test: torch.Tensor,
#                             labels_ref: torch.Tensor,
#                             top_k: int
#                             ) -> float:
#
#     _, indices = distances.topk(k=top_k, dim=1, largest=False)  # indices shape: [M x k]
#
#     y_pred = []
#     for i in range(top_k):
#         indices_at_k: torch.Tensor = indices[:, i]  # [M]
#         y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(dim=1)  # [M x 1]
#         y_pred.append(y_pred_at_k)
#
#     y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
#     labels_test = torch.hstack((labels_test,) * top_k)  # [M x k]
#
#     n_predictions: int = y_pred.shape[0]
#     n_true_predictions: int = ((y_pred == labels_test).sum(dim=1) > 0).sum().item()
#     topk_accuracy: float = n_true_predictions / n_predictions * 100
#     return topk_accuracy

# 0413

# NOT USE
# 0412
def get_poison_embeddings_from_dataloader(loader: DataLoader,
                                   model: nn.Module,
                                   device: torch.device,
                                   return_numpy_array=False,
                                   return_image_paths=False,
                                   delta=0.2,
                                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls_poi: List[torch.Tensor] = []
    labels_ls_poi: List[torch.Tensor] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)

        # 0412 poison images for embeddings
        images_ori = images.clone().detach()
        images_poi = images_ori + delta * 2 * (0.5 - torch.rand(images_ori.shape)).to(images_ori.device)
        embeddings_poi: torch.Tensor = model(images_poi)

        #embeddings: torch.Tensor = model(images)
        embeddings_ls_poi.append(embeddings_poi)
        labels_ls_poi.append(labels)

    embeddings_poi: torch.Tensor = torch.cat(embeddings_ls_poi, dim=0)  # shape: [N x embedding_size]
    labels_poi: torch.Tensor = torch.cat(labels_ls_poi, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings_poi = embeddings_poi.cpu().numpy()
        labels_poi = labels_poi.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings_poi, labels_poi, images_paths)

    return (embeddings_poi, labels_poi)

### NOT USE
def get_poi_emb_from_images(images_, labels_,
                                   model: nn.Module,
                                   device: torch.device, delta=0.1,
                                   return_numpy_array=False,
                                   return_image_paths=False,
                                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    # for images_, labels_ in loader:
    images: torch.Tensor = images_.to(device, non_blocking=True)
    labels: torch.Tensor = labels_.to(device, non_blocking=True)
    embeddings: torch.Tensor = model(images)

    emb_delta = embeddings + delta * 2 * (0.5 - torch.rand(embeddings.shape)).to(embeddings.device)

    embeddings_ls.append(emb_delta)
    labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.cpu().numpy()

    # if return_image_paths:
    #     images_paths: List[str] = []
    #     for path, _ in loader.dataset.samples:
    #         images_paths.append(path)
    #     return (embeddings, labels, images_paths)

    return (embeddings, labels)

# NOT USE
def imshow_cifar(images, labels, classes):
    kind='Cifar 10'
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize 没有这个会变成奇怪的黑色图
    if (torch.cuda.is_available() is False):
        np_img = img.numpy()
    else:
        np_img = img.cpu().numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(kind)
    plt.savefig('plot/' + kind + '_data.jpg')
    print( [classes[i.item()] for i in labels])
    plt.show()

# NOT USE
# torchattack
def get_adv_emb_from_dataloader(loader: DataLoader, model: nn.Module, attack,
                                device: torch.device, return_numpy_array=False,
                                return_image_paths=False,
                                ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    is_show = True
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)

        labels = torch.squeeze(labels, dim=1)
        adv_images = attack(images, labels)
        if is_show:
            imshow(adv_images,labels,'pgd', True)
            is_show = False

        embeddings: torch.Tensor = model(adv_images)
        embeddings_ls.append(embeddings)
        labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings, labels, images_paths)

    return (embeddings, labels)

# NOT USE
def get_poi_emb_from_dataloader(loader: DataLoader, model: nn.Module, device: torch.device, delta=0.1,
                                   return_numpy_array=False, return_image_paths=False,
                                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)
        embeddings: torch.Tensor = model(images)

        emb_delta = embeddings + delta * 2 * (0.5 - torch.rand(embeddings.shape)).to(embeddings.device)

        embeddings_ls.append(emb_delta)
        labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings, labels, images_paths)

    return (embeddings, labels)
