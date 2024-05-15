================
by Jawad Haider

# **00 - PyTorch Gradients**
------------------------------------------------------------------------
<center>
<a href=''>![Image](../../../assets/img/logo1.png)</a>
</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
------------------------------------------------------------------------

- <a href="#pytorch-gradients" id="toc-pytorch-gradients">PyTorch
  Gradients</a>
  - <a href="#autograd---automatic-differentiation"
    id="toc-autograd---automatic-differentiation">Autograd - Automatic
    Differentiation</a>
  - <a href="#back-propagation-on-one-step"
    id="toc-back-propagation-on-one-step">Back-propagation on one step</a>
  - <a href="#back-propagation-on-multiple-steps"
    id="toc-back-propagation-on-multiple-steps">Back-propagation on multiple
    steps</a>
  - <a href="#turn-off-tracking" id="toc-turn-off-tracking">Turn off
    tracking</a>

------------------------------------------------------------------------

# PyTorch Gradients

This section covers the PyTorch
<a href='https://pytorch.org/docs/stable/autograd.html'><strong><tt>autograd</tt></strong></a>
implementation of gradient descent. Tools include:  
\*
<a href='https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward'><tt><strong>torch.autograd.backward()</strong></tt></a>  
  \*<a href='https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad'><tt><strong>torch.autograd.grad()</strong></tt></a>
  

Before continuing in this section, be sure to watch the theory lectures
to understand the following concepts:  
\* Error functions (step andsigmoid)  
\* One-hot encoding  
\* Maximum likelihood  
\* Cross entropy(including multi-class cross entropy)  
\* Back propagation (backprop)

 
<div class="alert alert-info">

<h3>
Additional Resources:
</h3>
<strong>
<a href='https://pytorch.org/docs/stable/notes/autograd.html'>PyTorch
Notes:</a></strong>  <font color=black>Autograd mechanics</font>

</div>

## Autograd - Automatic Differentiation

In previous sections we created tensors and performed a variety of
operations on them, but we did nothing to store the sequence of
operations, or to apply the derivative of a completed function.

In this section we’ll introduce the concept of the <em>dynamic
computational graph</em> which is comprised of all the <em>Tensor</em>
objects in the network, as well as the <em>Functions</em> used to create
them. Note that only the input Tensors we create ourselves will not have
associated Function objects.

The PyTorch
<a href='https://pytorch.org/docs/stable/autograd.html'><strong><tt>autograd</tt></strong></a>
package provides automatic differentiation for all operations on
Tensors. This is because operations become attributes of the tensors
themselves. When a Tensor’s <tt>.requires_grad</tt> attribute is set to
True, it starts to track all operations on it. When an operation
finishes you can call <tt>.backward()</tt> and have all the gradients
computed automatically. The gradient for a tensor will be accumulated
into its <tt>.grad</tt> attribute.

Let’s see this in practice.

## Back-propagation on one step

We’ll start by applying a single polynomial function  
![y = f(x)](https://latex.codecogs.com/svg.latex?y%20%3D%20f%28x%29 "y = f(x)")
to tensor ![x](https://latex.codecogs.com/svg.latex?x "x"). Then we’ll
backprop and print the gradient
![\frac {dy} {dx}](https://latex.codecogs.com/svg.latex?%5Cfrac%20%7Bdy%7D%20%7Bdx%7D "\frac {dy} {dx}").

![\begin{split}Function:\quad y &= 2x^4 + x^3 + 3x^2 + 5x + 1 \\\\  Derivative:\quad y' &= 8x^3 + 3x^2 + 6x + 5\end{split}](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bsplit%7DFunction%3A%5Cquad%20y%20%26%3D%202x%5E4%20%2B%20x%5E3%20%2B%203x%5E2%20%2B%205x%20%2B%201%20%5C%5C%20%20Derivative%3A%5Cquad%20y%27%20%26%3D%208x%5E3%20%2B%203x%5E2%20%2B%206x%20%2B%205%5Cend%7Bsplit%7D "\begin{split}Function:\quad y &= 2x^4 + x^3 + 3x^2 + 5x + 1 \\  Derivative:\quad y' &= 8x^3 + 3x^2 + 6x + 5\end{split}")

#### Step 1. Perform standard imports

``` python
import torch
```

#### Step 2. Create a tensor with <tt>requires_grad</tt> set to True

This sets up computational tracking on the tensor.

``` python
x = torch.tensor(2.0, requires_grad=True)
```

#### Step 3. Define a function

``` python
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1

print(y)
```

    tensor(63., grad_fn=<AddBackward0>)

Since ![y](https://latex.codecogs.com/svg.latex?y "y") was created as a
result of an operation, it has an associated gradient function
accessible as <tt>y.grad_fn</tt><br> The calculation of
![y](https://latex.codecogs.com/svg.latex?y "y") is done as:<br>

![\quad y=2(2)^4+(2)^3+3(2)^2+5(2)+1 = 32+8+12+10+1 = 63](https://latex.codecogs.com/svg.latex?%5Cquad%20y%3D2%282%29%5E4%2B%282%29%5E3%2B3%282%29%5E2%2B5%282%29%2B1%20%3D%2032%2B8%2B12%2B10%2B1%20%3D%2063 "\quad y=2(2)^4+(2)^3+3(2)^2+5(2)+1 = 32+8+12+10+1 = 63")

This is the value of ![y](https://latex.codecogs.com/svg.latex?y "y")
when ![x=2](https://latex.codecogs.com/svg.latex?x%3D2 "x=2").

#### Step 4. Backprop

``` python
y.backward()
```

#### Step 5. Display the resulting gradient

``` python
print(x.grad)
```

    tensor(93.)

Note that <tt>x.grad</tt> is an attribute of tensor
![x](https://latex.codecogs.com/svg.latex?x "x"), so we don’t use
parentheses. The computation is the result of<br>

![\quad y'=8(2)^3+3(2)^2+6(2)+5 = 64+12+12+5 = 93](https://latex.codecogs.com/svg.latex?%5Cquad%20y%27%3D8%282%29%5E3%2B3%282%29%5E2%2B6%282%29%2B5%20%3D%2064%2B12%2B12%2B5%20%3D%2093 "\quad y'=8(2)^3+3(2)^2+6(2)+5 = 64+12+12+5 = 93")

This is the slope of the polynomial at the point
![(2,63)](https://latex.codecogs.com/svg.latex?%282%2C63%29 "(2,63)").

## Back-propagation on multiple steps

Now let’s do something more complex, involving layers
![y](https://latex.codecogs.com/svg.latex?y "y") and
![z](https://latex.codecogs.com/svg.latex?z "z") between
![x](https://latex.codecogs.com/svg.latex?x "x") and our output layer
![out](https://latex.codecogs.com/svg.latex?out "out"). \  

#### 1. Create
a tensor

``` python
x = torch.tensor([[1.,2,3],[3,2,1]], requires_grad=True)
print(x)
```

    tensor([[1., 2., 3.],
            [3., 2., 1.]], requires_grad=True)

#### 2. Create the first layer with ![y = 3x+2](https://latex.codecogs.com/svg.latex?y%20%3D%203x%2B2 "y = 3x+2")

``` python
y = 3*x + 2
print(y)
```

    tensor([[ 5.,  8., 11.],
            [11.,  8.,  5.]], grad_fn=<AddBackward0>)

#### 3. Create the second layer with ![z = 2y^2](https://latex.codecogs.com/svg.latex?z%20%3D%202y%5E2 "z = 2y^2")

``` python
z = 2*y**2
print(z)
```

    tensor([[ 50., 128., 242.],
            [242., 128.,  50.]], grad_fn=<MulBackward0>)

#### 4. Set the output to be the matrix mean

``` python
out = z.mean()
print(out)
```

    tensor(140., grad_fn=<MeanBackward1>)

#### 5. Now perform back-propagation to find the gradient of x w.r.t out

(If you haven’t seen it before, w.r.t. is an abbreviation of <em>with
respect to</em>)

``` python
out.backward()
print(x.grad)
```

    tensor([[10., 16., 22.],
            [22., 16., 10.]])

You should see a 2x3 matrix. If we call the final <tt>out</tt> tensor
“![o](https://latex.codecogs.com/svg.latex?o "o")”, we can calculate the
partial derivative of ![o](https://latex.codecogs.com/svg.latex?o "o")
with respect to ![x_i](https://latex.codecogs.com/svg.latex?x_i "x_i")
as follows:<br>

![o = \frac {1} {6}\sum\_{i=1}^{6} z_i](https://latex.codecogs.com/svg.latex?o%20%3D%20%5Cfrac%20%7B1%7D%20%7B6%7D%5Csum_%7Bi%3D1%7D%5E%7B6%7D%20z_i "o = \frac {1} {6}\sum_{i=1}^{6} z_i")<br>

![z_i = 2(y_i)^2 = 2(3x_i+2)^2](https://latex.codecogs.com/svg.latex?z_i%20%3D%202%28y_i%29%5E2%20%3D%202%283x_i%2B2%29%5E2 "z_i = 2(y_i)^2 = 2(3x_i+2)^2")<br>

To solve the derivative of
![z_i](https://latex.codecogs.com/svg.latex?z_i "z_i") we use the
<a href='https://en.wikipedia.org/wiki/Chain_rule'>chain rule</a>, where
the derivative of
![f(g(x)) = f'(g(x))g'(x)](https://latex.codecogs.com/svg.latex?f%28g%28x%29%29%20%3D%20f%27%28g%28x%29%29g%27%28x%29 "f(g(x)) = f'(g(x))g'(x)")<br>

In this case<br>

![\begin{split} f(g(x)) &= 2(g(x))^2, \quad &f'(g(x)) = 4g(x) \\\\ g(x) &= 3x+2, &g'(x) = 3 \\\\ \frac {dz} {dx} &= 4g(x)\times 3 &= 12(3x+2) \end{split}](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bsplit%7D%20f%28g%28x%29%29%20%26%3D%202%28g%28x%29%29%5E2%2C%20%5Cquad%20%26f%27%28g%28x%29%29%20%3D%204g%28x%29%20%5C%5C%20g%28x%29%20%26%3D%203x%2B2%2C%20%26g%27%28x%29%20%3D%203%20%5C%5C%20%5Cfrac%20%7Bdz%7D%20%7Bdx%7D%20%26%3D%204g%28x%29%5Ctimes%203%20%26%3D%2012%283x%2B2%29%20%5Cend%7Bsplit%7D "\begin{split} f(g(x)) &= 2(g(x))^2, \quad &f'(g(x)) = 4g(x) \\ g(x) &= 3x+2, &g'(x) = 3 \\ \frac {dz} {dx} &= 4g(x)\times 3 &= 12(3x+2) \end{split}")

Therefore,<br>

![\frac{\partial o}{\partial x_i} = \frac{1}{6}\times 12(3x+2)](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20o%7D%7B%5Cpartial%20x_i%7D%20%3D%20%5Cfrac%7B1%7D%7B6%7D%5Ctimes%2012%283x%2B2%29 "\frac{\partial o}{\partial x_i} = \frac{1}{6}\times 12(3x+2)")<br>

![\frac{\partial o}{\partial x_i}\bigr\rvert\_{x_i=1} = 2(3(1)+2) = 10](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20o%7D%7B%5Cpartial%20x_i%7D%5Cbigr%5Crvert_%7Bx_i%3D1%7D%20%3D%202%283%281%29%2B2%29%20%3D%2010 "\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = 2(3(1)+2) = 10")

![\frac{\partial o}{\partial x_i}\bigr\rvert\_{x_i=2} = 2(3(2)+2) = 16](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20o%7D%7B%5Cpartial%20x_i%7D%5Cbigr%5Crvert_%7Bx_i%3D2%7D%20%3D%202%283%282%29%2B2%29%20%3D%2016 "\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=2} = 2(3(2)+2) = 16")

![\frac{\partial o}{\partial x_i}\bigr\rvert\_{x_i=3} = 2(3(3)+2) = 22](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20o%7D%7B%5Cpartial%20x_i%7D%5Cbigr%5Crvert_%7Bx_i%3D3%7D%20%3D%202%283%283%29%2B2%29%20%3D%2022 "\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=3} = 2(3(3)+2) = 22")

## Turn off tracking

There may be times when we don’t want or need to track the computational
history.

You can reset a tensor’s <tt>requires_grad</tt> attribute in-place using
`.requires_grad_(True)` (or False) as needed.

When performing evaluations, it’s often helpful to wrap a set of
operations in `with torch.no_grad():`

A less-used method is to run `.detach()` on a tensor to prevent future
computations from being tracked. This can be handy when cloning a
tensor.

<div class="alert alert-info">

<strong>A NOTE ABOUT TENSORS AND VARIABLES:</strong> Prior to PyTorch
v0.4.0 (April 2018) Tensors (<tt>torch.Tensor</tt>) only held data, and
tracking history was reserved for the Variable wrapper
(<tt>torch.autograd.Variable</tt>). Since v0.4.0 tensors and variables
have merged, and tracking functionality is now available through the
<tt>requires_grad=True</tt> flag.

</div>
