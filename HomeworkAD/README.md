# LINMA2472 - Algorithms in data science
## Homework handout: Vectorized AD, Second-order AD and application to Transformers

### Part 1: Vectorized AD

Implement `reverse_vectorized.jl` so that it passes the tests in `test.jl`.
You should support computing the gradient of the functions tested in `test.jl` with a `Flatten` as input.
Create one `VectNode` for each component of the `Flatten`, instead of one `Node` for each scalar (as was done in the lab).
More precisely, while the AD of the lab only supports node of scalar value, your AD should support nodes for which the value can be arrays as well.
Your AD implementation does not need to support any case that is not covered by the tests.
The output `VectReverse.gradient` should match (up to reasonable accuracy, you can't fight floating point rounding errors) the output of `Forward.gradient` where `Forward` is the module defined in the solution of LabAD.

Benchmark your implementation and comment on the result, e.g.,
What is the bottleneck in the computation of the gradient ?
Does this match your expectation/complexity analysis ?
How would the memory and time requirement evolve with the size of the neural network ?
How could this be reduced ?

### Part 2: Second-order AD

Add support for the `hvp` computing a hessian-vector product in `reverse_vectorized.jl`.
The output of `VectReverse.hvp` should match the output of `Forward.hvp` where `Forward` is the module
defined in the solution of LabAD.

Benchmark your implementation and comment on the result, e.g.,
What is the bottleneck in the computation of the hessian-vector product ?
Does this match your expectation/complexity analysis ?
How would the memory and time requirement evolve with the size of the neural network ?
How could this be reduced ?

How can this new feature be used to train a neural network ?
Experiment training a neural network with and without exploiting `hvp`.

> [!TIP]
> For this, you are allowed to reuse the example from LabAD but you can also consider other neural network training problems.

Comment on the results.

### Part 3: Transformers

Implement a transformer and train it using `VectReverse`.
For this, you may need to add support for more operators in your `VectReverse` to be able to differentiate through your transformer.
Train your transformer on some dataset of your choice and show us the result!

Benchmark the gradient computation and the evolution of the loss training in training and comment on your observations.

> [!NOTE]
> We allow your `VectReverse.hvp` to lag behind `VectReverse.gradient` and not support these additional operators.
> In other words, the `VectReverse.hvp` function should support all operators used by the tests but not necessarily
> all operators used by your transformer implementation.

> [!TIP]
> [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) shows how to train a transformer from scratch
> to write like Shakespeare. Feel free to get inspiration from this, e.g., by using the Shakespeare dataset,
> the transformer sizes they use, etc... Of course, you may not use Pytorch like shown in that course, you need to use
> your own `VectReverse` AD !

> [!TIP]
> To speed up the computation, you can use the GPU of your computer or of [the CECI cluster](https://github.com/blegat/LINMA2710?tab=readme-ov-file#ceci-cluster).
> To accelerate AD on GPU, simply convert the `Array`'s to [`CUDA.CuArray`'s (for NVIDIA GPUs)](https://github.com/JuliaGPU/CUDA.jl/) or
> [`AMDGPU.ROCArray`'s (for AMD GPUs)](https://github.com/JuliaGPU/AMDGPU.jl).
> That will transfer the array to your GPU memory, all operations will then be done on the GPU until you convert it back to an `Array`.
> This means that you should be able to differentiate on the GPU without the need to change the code of `VectReverse`.
> Note that even though GPUs support computation with double precision (i.e., `Float64`), they are much faster with single precision
> (i.e., `Float32`) so it may be appropriate to use `Float32` instead of `Float64`.

### Practical information

Deadline, oral presentation, submission information, TBA

> [!TIP]
> Any question ? Ask us on the Moodle forum!

### Further readings and references

* *Evaluating Derivatives*, A. Griewank and A. Walther
* *The Elements of Differentiable Programming*, M. Blondel, V. Roulet
* 3Blue1Brown, https://www.3blue1brown.com/?v=chain-rule-and-product-rule, https://www.3blue1brown.com/?v=neural-networks, https://www.3blue1brown.com/lessons/backpropagation#title
