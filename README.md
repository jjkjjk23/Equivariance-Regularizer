# Equivariance-Regularizer
A regularization term for Pytorch that pushes the model to be more equivariant with respect to a given semigroup action. 

In particular, the user specifies a list of quadruples (f0, f1, epsilon, lambda_eq) where f0 is a transformation to be applied to the outputs of the neural network, f1 is a transformation to be applied to the inputs, epsilon is a threshold value which we will explain, and lambda_eq is a weight constant that will scale the regularization term. 

For an example consider a neural net that classifies images by whether or not they contain a dog. Then good examples for f1 would be shifting the image by a pixel in some direction, or rotating the image by a small angle. The corresponding f0 transformations would be the identity function because these minor transformations should not affect whether the image contains a dog or not. However, if the image is shifted too far it may become impossible to identify the dog. As we will see this is the purpose of the epsilon value.

Given a neural net called "model", a distance function "dist" (e.g. for regression tasks the l2 distance), one quadruple (f0,f1,epsilon), and an input point v, we define the equivariance error of the model at v as dist(f0(model(v)), model(f1(v))). Similarly the global equivariance error which we will call equivariance_error is the integral of this quantity as v varies over the input space of the neural net, usually a hypercube or the unit sphere. In practice the global equivariance error will be approximated by a Monte Carlo integral. Then the full regularization term is lambda_eq*max(0, equivariance_error-epsilon).

We see that if the equivariance error is exactly zero then the model is truly equivariant with respect to the transformations. For example for the dog classification example this means that the model will give the exact same answer for a shifted image as it did for the original image. If the equivariance error is small but not 0, this allows the answer to change if the image is shifted too far. 


