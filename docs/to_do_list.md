## Live Document with Possible Next Steps

***

- [x] Split data into training and test subsets
- [x] Write scripts with Keras functions to load the datasets and perform online data augmentation
- [x] Calculate the image complexity of each image on all dataset as measured by the MSE between the original image and its JPEG compressed version
- [x] Design a CAE architecture with appropriate complexity and use Bayesian search to optimize hyperparameters (and possibly depth)
- [x] Calculate the reconstruction error of each image on the test set using the optimized CAE
- [ ] Experiments with varying autoencoder depth
- [ ] Analyse if the CAE is overfitting during the hyperparameter optimization for very small datasets
- - - [ ] Consider different strategies for building the architecture
- - - [ ] Consider collecting more samples to smaller datasets
- [ ] Using a general purpose CAE for baseline comparison
- [ ] Using a general purpose dataset for baseline comparison
- [ ] Investigate dataset homegeneity in terms of resolution and file size