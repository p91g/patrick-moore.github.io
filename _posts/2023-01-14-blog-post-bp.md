---
title: 'Deep Learning - What is backpropogation?'
date: 2023-01-14
permalink: /posts/2023/01/blog-post-1/
tags:
  - cool posts
  - category1
  - category2
---
 <br/><img src='https://p91g.github.io/patrick-moore.github.io/images/bp.jpg'>{: .align-right width="300px"}
- In this post I give brief summary of what the backpropogation algorithm is. 
- Backpropogation is fundamental to the optimisation process for neural networks! It is somewhat intimidating until you delve into the details.

Definition
===
It is a method for computing the gradients of parameters in a neural network by recursively applying the chain rule. These gradients are used to update the weights and biases of the network during training, in order to minimise the loss function. In other words, propagating the error (i.e. gradients) backward through the network to be updated.

_The steps of the algorithm are:_

Forward pass
---
Starting at the inputs going toward the output, intermediate operations are calculated through the network to calculate final predictions which are compared to the target values defined as the ‘loss’. This process is typically performed using matrix multiplication and activation functions. The values of the variables and parameters are stored for use in the backward pass.

Backward pass
------
Beginning at the output (the loss) of the network, moving backward, the derivative of the loss is calculated with respect to each intermediate output operation, used to calculate the derivative with respect to the inputs.

The derivative of each variable tells us the sensitivity of its contribution to changing the slope of a given function.

The chain rule is used when the derivative is taken from multiple composed functions, as is the case with neural networks. The composed functions are substituted into intermediate variables so that they can be 'chained', i.e. multiplied together. This makes computation easier as each derivative can be handled in isolation. For example, the derivative of the loss with respect to a weight in a particular layer is multiplied by the derivatives of the intermediate operations in that layer. 

For a multivariable function, as is the case for multi-layer neural networks, the multivariable chain rule is used, which equates to taking the partial derivative of each variable and storing it in a gradient vector and dotting it with the derivative of the inner functions. Note, that the dot-product both sums and multiplies, meaning we add up the partials for each variable. 

Because the inner functions are also vectors, the output is a matrix, which can be generalised to a handy matrix called the 'Jacobian'. Each row is a vector valued function and each column is a variable we are taking the derivative with respect to. The Jacobian tells us the relationship between each input element and each output element, so that we know how much the function will change for a small change to the inputs. 

Multiple vector valued functions are stacked together into one matrix and can be efficiently chained together with other jacobians, representing the layers of a neural network. For example, the Jacobian can be used to calculate the derivative of the output layer with respect to the input of a previous layer, by multiplying the Jacobian for the current layer with the derivative of the activation function of the previous layer.

Note: 
- Here the derivative is taken w.r.t X, but in practice we would only compute the gradient of W and b. 
- The partial derivatives of the functional form of a neuron activation - ie. L = ReLu (W X +b) and the loss function, such as cross-entropy or RMSE, is not included as several derivation steps are required in matrix form. For example, the derivative identities of a matrix w.r.t. a vector, derivative of a vector w.r.t. a scalar, and the inverses.



