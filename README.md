# Vulnerable_Metric_Learning
Use PGD attack to poison a deep metric learning model based on triple loss

'train_v3.py' : train poison and clean models
'models/lenet-triplet.py': modified lenet with triple loss
'attack/attack_pgd.py': modified pgd attack from torchattacks
'datasets/mnist': load mnist dataset
