# Adversarial-Examples


## About The Project

1) Implementation of Fast Gradient Sign Method (FGSM) from `Explaining and Harnessing Adversarial Examples` paper:

        https://arxiv.org/pdf/1412.6572.pdf

2) Using previuosly trained model (ResNet-50) (which was trained on Stanford Online Products Dataset (SOP) for classification task) for FGSM.

3) Getting adversarial examples for pictures from the list (filenames.txt).

4) Finding such ε that the neural network copes with the classifications the worst, but at the same time there is no visible noise appears in the picture.

5) Changing pictures and saving them for review.


## Getting Started

File to run:

    train/main.py
  
  
## Additional Information

Parameters can be changes in:

        configs/config_improved_model.py 
        

