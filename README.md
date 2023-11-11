# Equivariance-Regularizer
~~~
pip install equivariance-regularizer==0.1.0
~~~
A regularization term for Pytorch that pushes the model to be more equivariant with respect to a given semigroup action. 

### CLASS: EquivarianceError(model, shape, transforms, dist=2, n=1, num_funcs=1, bounds=[0,1])
1. model is a PyTorch module, 
2. shape is the input shape for tensors in the neural net. This SHOULD NOT include a batch dimension.
3. transforms is a list of tuples (f_out, f_in, epsilon, lambda_eq) where:
    1. f_out is a transformation to be applied to the outputs of the neural network.
    2. f_in is a transformation to be applied to the inputs
    3. epsilon is a threshold value.
    4. lambda_eq is a weight constant that will scale the regularization term. 
    5. If f_out equals f_in, f_in should be omitted, so that the tuple is (f_out, epsilon, lambda_eq).
4. dist is a distance function, so it should take in two tensors with the shape (batch, output_shape) and return either a scalar or a tensor of size (batch). If the output is not a scalar, the values of that tensor will be averaged.  The default is the l2 distance but other examples are given in the documentation.
The user could also pass in an integer p (or the string 'inf'), which will give the lp distance, or one of the strings "cross_entropy" or "cross_entropy_logits".
5. n is the sample size for the Monte Carlo integral.
6. num_funcs is the number of functions considered each time the equivariance error is calculated. A low value for num_funcs (as well as n) gives a coarser approximation to the true equivariance error, but is much better for performance and is still effective in practice.
7. bounds is the bounds for the Monte Carlo integral. It should be a tensor of shape (shape,2) where bounds[...,0] is a tensor of lower bounds and bounds[...,1] of upper bounds. Defaults to the hypercube of tensors with entries in the interval [0,1]. This is appropriate if the model takes tensors with positive entries and begins by unit normalization of input tensors so that the domain of integration is really the positive orthant of the unit sphere. This is the case for many tasks relating to images and audio. 
If the bounds provided have a different shape they will be broadcasted to the above shape. Also if given as python lists or numpy arrays they will be converted torch tensors. Therefore the user could give bounds = [0,1] and this would give the same as the default.

The objects in this class are callable. When called with no arguments the equivariance error will be calculated on a random sample of size n points inside the given bounds. If called with a single tensor argument, the equivariance error will be calculated at that point. Natural inputs would be elements of the training set or vectors that are randomly generated using a different scheme than our default. 

## Implementation
~~~python
from equivariance_regularizer import EquivarianceRegularizer
class Model(torch.nn.Module):
    ...
    def __init__(transforms, dist, n, num_funcs):
        ...
    def training_step(self):
        ...
        loss += equivariance_error()
        ...
model = Model()
equivariance_error = EquivarianceRegularizer(model, shape, transforms, dist, n, num_funcs, bounds)
~~~

Note that this seems backwards because equivariance_error appears in the training step which is before equivariance_error is defined, but this is no problem as long as equivariance_error is not referred to while initializing the model. It is necessary for the model to be defined before equivariance_error is.

Also, it is recommended not to initialize EquivarianceRegularizer inside of the model with a call like 
~~~python
EquivarianceRegularizer(self, shape, transforms, dist, n, num_funcs, bounds)
~~~
This can cause an infinite recursion error.

In practice, it can be very memory intensive to calculate the equivariance error along with the loss. If this is an issue it is advisable to alternate between a standard training step and a training step that only does a backwards pass of the equivariance error.

## What it does:

Given a neural net called "model", a distance function "dist" (e.g. for regression tasks the l2 distance), one tuple (f_out,f_in,epsilon, lambda_eq), and an input point v, we define the equivariance error of the model at v as 

~~~python
dist(f_out(model(v)), model(f_in(v))). 
~~~

Similarly the global equivariance error which we will call "error" is the integral of this quantity as v varies over the input space of the neural net, usually a hypercube or the unit sphere. In practice the global equivariance error will be approximated by a Monte Carlo integral. Then the full regularization term is 

~~~python
lambda_eq*max(0, error-epsilon).
~~~

The model is called equivariant with respect to f_in and f_out if this quantity is 0.

## Examples

For an example consider a neural net that classifies images by whether or not they contain a dog. Then f_in could be shifting the image by a pixel in some direction, or rotating the image by a small angle. The corresponding f_out transformations would be the identity function because these minor transformations should not affect whether the image contains a dog or not. However, if the image is shifted too far it may become impossible to identify the dog. This is the purpose of the epsilon value: the model will not try to minimize the equivariance error as long as it is below epsilon.

A slightly different example would be a neural net that performs image segmentation, for instance one that outlines the dog in a given picture. Given a transform f such as shifting, rotating, warping, etc, this neural net should be equivariant with f_in = f_out = f, and epsilon = 0.

## Distance Functions
As stated above, the l2 distances and (similarly lp for any p) are appropriate for regression tasks. 

For classification, when the loss function is cross entropy, an appropriate distance function is the distance between the log likelihoods, however one should be careful about the way the transformation f_out introduces components with zeros. 

So if the model has no softmax layer and is outputting logits, then it would be appropriate to use "cross_entropy_logits" as the dist argument, and the dist function is essentially (ignoring that the tensors are batched)

~~~python
def cross_entropy_logits(tensor1, tensor2):
    tensor1 = torch.nn.functional.log_softmax(tensor1, dim=-1)
    tensor2 = torch.nn.functional.log_softmax(tensor2, dim=-1)
    return torch.dist(tensor1,tensor2,2)
~~~
Note that in this case, the output transforms are applied to the logits, not the Softmax. This can be good, but if it is not desired it is better to end the model with a softmax layer and use dist = "cross_entropy".

If the model does end with a softmax layer, set dist = "cross_entropy" and the distance function will use torch.log instead of log_softmax.

## Random Transforms

Many transforms used for data augmentation are random, e.g. rotate by a random angle. If the input transformation is random like this and the output function doesn't depend on the randomly chosen parameter (for instance if the output transformation is the identity transformation and does nothing no matter what angle the input is rotated by) then there is no need to modify any of the above setup. However, for tasks such as image segmentation, where the input transformation usually equals the output transformation, we need to be careful. In this case, the transform should be input as (f_out, epsilon, lambda_eq), and the package will make sure that the exact same random transformation is applied to both the input and output.




