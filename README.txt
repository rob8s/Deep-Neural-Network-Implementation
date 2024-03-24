Command Line Args

[-v] - verbose mode
-train_feat TRAIN_FEAT_FN - train data predictors - txt file
-train_target TRAIN_TARGET_FN - train data targets - txt file
-dev_feat DEV_FEAT_FN - dev data predictors - txt file
-dev_target DEV_TARGET_FN - dev data targets - txt file
-epochs EPOCHS - number of epochs to train for - any int
-learnrate LEARNRATE - backprop spec - float
-nunits NUM_HIDDEN_UNITS - layer size - int
-type PROBLEM_MODE - R = Regression, C = Classification
-hidden_act HIDDEN_UNIT_ACTIVATION - nonlinear activation func - relu, sig, tanh
-init_range INIT_RANGE - random itialized weights and biases - float
[-num_classes C] - dimension of targets - int
[-mb MINIBATCH_SIZE] - minibatch size for train - int
[-nlayers NUM_HIDDEN_LAYERS] - number of desired layers - int

Deep Neural Network implimentation, designed for split test and dev data(for eval). Trains itself with Backpropogation implimentation while supporting any size minibatches. Data is required to be in numerical form, if Classification, program will onehot if required. Saves the best performing model, but does not have early stopping implimented. If verbose mode, will display Train and Dev loss after each pass through the model, while not in verbose mode will display Train and Dev loss after each epoch. If target dimension is greater then 1, num_classes is required for program to function. 



