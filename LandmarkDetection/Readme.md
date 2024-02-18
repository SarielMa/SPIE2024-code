
# A general approach to improve adversarial robustness of DNNs for medical image segmentation and detection

To install the basic multi-task unet, and download the Cephalometric dataset, please refer to https://github.com/qsyao/attack_landmark_detection.

train.py trains the multi-task unet with original settings in the paper, Miss the Point: Targeted Adversarial Attack on Multiple Landmark Detection;

train.py trains the model without adversarial training;

train_AT.py trains the model with vanilla adversarial training;

train_Ours_2zscore.py trains the model with our proposed method; 

train_TE_L2.py trains the model with TEAT adversarial training method;

train_TRADES_L2.py trains the model with TEAT adversarial training method.
