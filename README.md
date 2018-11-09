# adversarial_perturbation_project
To assess the robustness of different neural network models to adversarial perturbations.

This repository contains three neural network models to train and test on MNIST. 

1. Two-layer perceptron (MLP)
2. Locally Competative Algorithm (LCA) with a classifier 
3. Directed LCA (DLCA) (with a feedback term) with a classifier 

All of the models are located in the models folder.

The repository also contains a adversarial generator based off of the fast gradient sign method
(Goodfellow et. al. 2015).

# Running Instuctions:

1. Run each model script in the models folder. Set model parameters in the script. (epochs, learningRate, etc.)
2. Weights will be saved for each model in the results forlder in a new folder named weights_bias.
3. Run the perturbation_generator.py script in the main folder. Set parameters in the script (num_examples, confidence_threshold, etc.)
4. When running, it will prompt you to choose a pre-trained weight_bias.
5. Once a weight_bias file is chosen, it will record metrics (MSE, PSNR) on generated adversarial examples.
6. It will then prompt you to name the file where the metrics for that model will be saved in a new folder results/adversarial_results.
7. Repeat for each pretrained model's weight_bias
8. Run the model_analysis.py script in the main folder.
9. Follow steps 4-7. Test metrics will be saved in results/test_results folder.
10. In the results folder, run plot_results.py to plot the testing and adversarial results. Set which files to choose within the script.

