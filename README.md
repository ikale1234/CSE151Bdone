# Ishaan Kale CSE 151B Project
Hi, here are some quick notes about my code in case you want to reproduce results.

This is the same structure as the starter repo. I have modified main.py, src/models.py, and some config files (for tuning epochs and learning rate, and other hyperparams).

To run my model, which is a ConvLSTM, please remain the "conv_lstm" in the training config. Tune the learning rate and number of epochs in their respective config files. For the rest of the hyperparameters except for sequence length, please modify them within models.py (in the init function of ConvLSTM module). I found using the configs pretty cumbersome. For tuning the sequence length, please do it in main.py, in the init function of the ClimateDataset module. Manipulation has to occur in data loading as well.

Otherwise, run it the same how you would for the starter repo (executing main.py).
