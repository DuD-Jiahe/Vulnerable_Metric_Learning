# Vulnerable_Metric_Learning
Use PGD attack to poison a deep metric learning model based on triple loss

'train_v3.py' : train poison and clean models

'models/lenet-triplet.py': modified lenet with triple loss

'attack/attack_pgd.py': modified pgd attack from torchattacks

'datasets/mnist': load mnist dataset

In this project, a poisoning attack is proposed specifically for supervised deep metric 
learning. First, a method based on triplet loss function is introduced to generate poisoned 
samples, then an objective function is designed for poisoning attacks. Since metric learning 
outputs representations in the feature space rather than confidence for each category, a method 
is proposed to make predictions based on the K-nearest neighbor algorithm. Experiments
show that the contaminated model has good performance on benign samples the same as the 
clean model, but low on the poisoned ones. Additionally, the tiny perturbations added to the 
original data make the attack difficult to detect.
