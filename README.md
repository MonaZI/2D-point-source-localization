# 2D-point-source-localization
Matlab and Python implementations for the following papers.

## Geometric Invariants for Sparse Unknown View Tomography
Authors: Mona Zehni, Shuai Huang, Ivan Dokmanic, Zhizhen Zhao

In this paper, we study a 2D tomography problem for point source models with random unknown view angles. 
Rather than recovering the projection angles, we reconstruct the model through a set of rotation-invariant features that are estimated from the projection data. For a point source model, we show that these features reveal geometric information about the model such as the radial and pairwise distances. This establishes a connection between unknown view tomography and unassigned distance geometry problem (uDGP). We propose new methods to extract the distances and approximate the pairwise distance distribution of the underlying points. We then use the recovered distribution to estimate the locations of the points through constrained non-convex optimization. Our simulation results verify the robustness of our point source reconstruction pipeline to noise and error in the estimation of the features.

Link to paper: https://arxiv.org/pdf/1811.09940.pdf

## Distance retrieval from unknown view tomography of 2D point sources
Authors: Mona Zehni, Shuai Huang, Ivan Dokmanic, Zhizhen Zhao

In this paper, we study a 2D tomography problem with random and unknown projection angles for a point source model. Specifically, we target recovering geometry information, i.e. the radial and pairwise distances of the underlying point source model. For this purpose, we introduce a set of rotation-invariant features that are estimated from the projection data. We further show these features are functions of the radial and pairwise distances of the point source model. By extracting the distances from the features, we gain insight into the geometry of the unknown point source model. This geometry information can be used later on to reconstruct the point source model. The simulation results verify the robustness of our method in presence of noise and errors in the estimation of the features.

Link to paper: https://bit.ly/2YqKCRb

##Files
Run main.m for a quick example of generating the features for Gaussian source model
Run radial_prony.m and pairwise_prony.m for extracting radial and pairwise distances
Python implementations of the same functionalities are provided in ./python directory
A jupyter notebook with a demo example is provided in ./python/cryoPDF.ipynb
