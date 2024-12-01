
.. _contribs:

Contributions/Problems
=============================

Thank you for your interest in this project! 
As the sole maintainer, I am currently not accepting pull requests or code contributions, 
but I will reviewing:
    * bugs, or serious flaw reports, 
    * feature requests,
    * documentation suggestions,
    * and any other suggestions.

**If you have feature requests**

Please note that ``pygrad`` is a lightweight library with few dependencies. 
It should also be simple and easy to use. I want to keep it this way.
I am not looking to replace PyTorch or Tensorflow.

**How to Submit an Issue:**
    * Create issue: https://github.com/baubels/pygrad/issues.
    * Include as much detail as possible.
    * Check existing issues to ensure you don't submit a duplicate.
    * For bug reports: include steps to reproduce, expected behaviour, and actual behaviour.
    * For feature requests: describe the feature in detail and how it would benefit the project.
    * For suggestions: describe the suggestion, and why it might be relevant.

**Gradient computations are wrong and related issues:**

Some libraries will compute different gradients from one another for the same object.

For example, if you had a single Tensor A that produces two Tensors B and C, 
gradient contributions during backpropagation from B and C should be added. 
However, Pytorch has the convention of averaging gradient contributions.

For purposes of being able to test the correctness of ``pygrad`` more easily
and offer better cross-compatibility with ``Pytorch``, it was determined that 
``pygrad`` would obey the conventions ``Pytorch`` has when doing backpropagation.

If you are receiving unexpected gradients, please check if you are getting the same issue with ``Pytorch``
first before submitting an issue.

It's also been difficult to figure out how ``Conv2D`` in Pytorch is implemented when determining gradients
specifically when calling ``.backward()`` more than once. For this reason, ``pygrad`` will output different gradients
than might be expected here.

**What Happens After You Submit an Issue:**

I will review and implement if accepted.
If there are discussions, implementation decisions will wait until discussions are resolved.
