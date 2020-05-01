# Unsupervised Data Augmentation PyTorch
 Unsupervised Data Augmentation for Consistency training implemented in Pytorch 

# Goal

A fundamental weakness of deep learning is that it typically requires a lot of labeled data to work well. Semi-supervised learning (SSL)
one of the most promising paradigms to leverage unlabeled data to address this weakness. (https://ai.googleblog.com/2019/07/advancing-semi-supervised-learning-with.html). Unsupervised Data Augmentation (UDA)
(https://arxiv.org/abs/1904.12848), recently proposed technique, popularized the idea to reuse th e data augmentation techniques known from supervised learning for the unlabeled data in the sem i-supervised setting. The intuition behind UDA
is that even if we do not know the desired neural output for an unlabeled data,
we can expect the neural network to produce an output on the augmented (e.g.: horizontal flipped) data which is the augmented version of the neural output for the non-augmented data. 

The candidate should start with the classification task (CIFAR-10, SVHN), where the output the neural network do not have to be augmented. The candidate should evaluate the effect of augmentation techniques available from RandAugment while varying size of labeled data is assumed to be available. Which augmentations are the best?

# Optional

An optional advanced part of the assignment is to try out the UDA technique for semantic segmentation and/or 2D object detection on a lightweight dataset. [E.g. Cityscapes dataset.](https://www.cityscapes-dataset.com/) How much gain can be achieved with UDA in the early phase of evelopment, when only a small amount of labeled data is available? Which augmentations are the best?

# Code
## Examples

| Name | Valid Loss | Valid Acc | Reference |
| ---------------|------------|---------- | ------- |
| UDA - 4000 - 46000 | 47.32| 86.57 | ~86 |
| Supervised All|32.2| 90.12 | - |
| Supervised All w s. Aug|19.83| 93.97| 94.08 |
| Supervised 4000| 82.02 | 80.15 | - |



## Run
   `python UDA.py` for semi-supervised UDA implemenetation with 46000 unlabelled - 4000 labelled data
   
   `python supervised.py` for fully supervised with 50000 labelled data -- for baseline
   
   `python supervised.py --limit 4000` for fully supervised with 4000 labelled data -- for baseline

### Tensorboard
Results are logged in `tensorboard`

Run `tensorboard --logdir FOLDER_NAME` to see results


# References

Using code from [vdef-5 UDA Pytorch](https://github.com/vfdev-5/UDA-pytorch/tree/740a6740eab29f9b916ffa62c9baf4bec092c0d2)
Using code from [autoaugment](https://github.com/tensorflow/models/blob/master/research/autoaugment/)