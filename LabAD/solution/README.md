# LINMA2472 - Algorithms in data science
## AD lab solution: First-order scalar AD, forward and reverse

### Code files
* `forward.jl` : classic forward.
* `reverse_simple.jl`: simple reverse. Simply constructs a DAG for the whole expression (including constant parts) and computes derivatives with respect to all leaves (variables + constants).
* `reverse_scalar.jl`: better scalar reverse. Constructs Nodes only for variables and operations on variables (no more for constant parts) and computes derivatives with respect to variables. Advantage is that we do not differentiate with respect to constants anymore and do not construct constant parts of the DAG; Disadvantage is that we have more if-elses, since we have more symbols (e.g. additional symbols to distinguish cst * var from var * var). Moreover, we need to if-else in the forward *and* backward passes.
* `reverse_jacstoring.jl`: other scalar reverse. Tries to improve on the `reverse_scalar.jl` version via the following idea: We note that in this scalar case, we can store the local Jacobians (cheap to store) on the forward pass, so as to just have to multiply the local Jacobians on the way back (no need to if-else on the way back anymore). Advantage is that the backward pass is expected to be faster; Disadvantage is we have to store local Jacobians (which we do not want to store entirely if vect/mat/tensors, for redundancy reasons).
* `train.jl`: implements gradient descent based on our AD system.
* `models.jl`: defines the different activation functions and loss functions + some utils.
* `lab_forward.jl`: quickly tests the forward version and various activation functions.
* `lab_reverse.jl`: quickly tests the different reverse versions and various activation functions.

### Further readings and references
* *Evaluating Derivatives*, A. Griewank and A. Walther
* *The Elements of Differentiable Programming*, M. Blondel, V. Roulet
* 3Blue1Brown, https://www.3blue1brown.com/?v=chain-rule-and-product-rule, https://www.3blue1brown.com/?v=neural-networks, https://www.3blue1brown.com/lessons/backpropagation#title 