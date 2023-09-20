# Equivariance-Regularizer
A regularization term for Pytorch that pushes the model to be more equivariant with respect to a given semigroup action. 

### CLASS: EquivarianceError(model, transforms, dist=torch.dist, n=1, num_funcs=1)
1. model is a PyTorch module, 
2. transforms is a list of tuples (f_out, f_in, epsilon, lambda_eq) where:
    1. f_out is a transformation to be applied to the outputs of the neural network.
    2. f_in is a transformation to be applied to the inputs
    3. epsilon is a threshold value.
    4. lambda_eq is a weight constant that will scale the regularization term. 
3. dist is a distance function, so it must take in two tensors with the same shape and return a scalar. The default is the l2 distance but other examples are given in the documentation.
The user could also pass in an integer p (or inf), which will give the lp distance, or one of the strings "cross_entropy" or "cross_entropy_logits".
4. n is the sample size for the Monte Carlo integral.
5. num_funcs is the number of functions considered each time the equivariance error is calculated. A low value for num_funcs (as well as n) gives a coarser approximation to the true equivariance error, but is much better for performance and is still effective in practice.

## Implementation
~~~python
class Model(torch.nn.Module):
    ...
    def __init__(transforms, dist, n, num_funcs):
        ...
        self.equivariance_error = EquivarianceError(self, transforms, dist, n, num_funcs)
    def training_step(self):
        ...
        loss += self.equivariance_error()
~~~

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

So if the model has no softmax layer and is outputting logits, then it would be appropriate to use "cross_entropy_logits" as the dist argument, and the dist function is essentially

~~~python
def cross_entropy_logits(tensor1, tensor2):
    tensor1 = torch.nn.functional.log_softmax(tensor1, dim=-1)
    tensor2 = torch.nn.functional.log_softmax(tensor2, dim=-1)
    return torch.dist(tensor1,tensor2,2)
~~~

If the model does end with a softmax layer, set dist = "cross_entropy" and then we will use torch.log instead of log_softmax.






