## A general approach to improve adversarial robustness of DNNs for medical image segmentation and detection

To install the basic version of nnUnet and download the datasets, Heart, Hippocampus and Prostate MRI, please refer to https://github.com/MIC-DKFZ/nnUNet 

Remember to modify the path in the nnUnet/nnunet/paths.py

the training and evaluation modules are in nnUNet/nnunet/run/:

run_training.py to train the original nnunet model without adversarial training;

run_Ours_training.py to train the nnunet model with our proposed defense method;

run_AT_training.py to train the nnunet model with vanilla adversarial training method.




