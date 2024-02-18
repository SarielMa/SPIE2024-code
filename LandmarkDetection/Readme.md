
# A general approach to improve adversarial robustness of DNNs for medical image segmentation and detection

To install the basic multi-task unet, and download the Cephalometric dataset, please refer to https://github.com/qsyao/attack_landmark_detection.

train.py trains the multi-task unet with original settings in the paper, Miss the Point: Targeted Adversarial Attack on Multiple Landmark Detection;

train_with_dice_loss.py trains the model with dice loss;

train_PGD_dice.py trains the model with vanilla adversarial training;

train_Ours_2zscore.py trains the model with our proposed method 
train_TE.py trains the model with TEAT adversarial training method.
