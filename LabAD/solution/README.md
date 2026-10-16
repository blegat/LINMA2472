# LINMA2472 — Algorithms in data science

## AD lab solution: first-order scalar AD

All reverse-mode variants reuse `CGE.Node` for the
expression graph and `CGE.topological_order` for its traversal.
They differ in what they record while building the graph and in how they
perform the backward pass.

### Code files

* `forward.jl`: classic forward mode using dual numbers.
* `reverse_simple.jl`: the direct reverse-mode implementation used in the
  handout. Every operand, including a numerical constant, becomes an
  expression node. `CGE.pullback!` dispatches each operation to
  the corresponding scalar propagation method in this file and
  computes derivatives for every leaf, even though only derivatives with
  respect to the input-variable leaves are ultimately returned. This version
  has the simplest graph construction and backward rules.
* `reverse_ifelse.jl`: avoids creating nodes for constant operands whenever
  possible. Operations such as `variable * constant` consequently have a
  different arity from `variable * variable`. The graph is smaller and does
  not propagate derivatives into those constants, but both graph construction
  and the reverse rules require more method signatures. Multiplication at a zero-valued
  variable is the notable corner case: the constant cannot be recovered by
  dividing the result by the input, so that constant is retained as a node.
* `reverse_jacstoring.jl`: also avoids constant operand nodes, and computes and
  stores each scalar local Jacobian during the forward pass. Its backward pass
  is therefore generic: it only multiplies the accumulated derivative by the
  stored local derivatives. Its node-level `CGE.pullback!` method
  bypasses the package's operation switch. This can
  make the backward pass faster, at the cost of additional storage. Storing
  complete local Jacobians does not extend efficiently to vector, matrix, or
  tensor operations, where structured pullback rules avoid substantial
  redundancy.
* `lab_reverse.jl`: compares the three reverse variants with forward mode on
  the lab examples and includes the performance exercises.

`CGE.backward!` accepts `backward!(output, topo)`, where `topo`
is a previously computed topological order. This numerical reverse sweep is
allocation-free and is useful when differentiating the same recorded graph
repeatedly. The convenient `backward!(output)` method computes the topological
order first; that graph traversal allocates a visited set and an ordering
vector. Each metadata type only implements `seed_metadata!` to initialize the
sweep, with a Boolean indicating whether the node is the output.

### Further reading

* *Evaluating Derivatives*, A. Griewank and A. Walther.
* *The Elements of Differentiable Programming*, M. Blondel and V. Roulet.
* 3Blue1Brown: [The chain rule](https://www.3blue1brown.com/?v=chain-rule-and-product-rule),
  [neural networks](https://www.3blue1brown.com/?v=neural-networks), and
  [backpropagation](https://www.3blue1brown.com/lessons/backpropagation#title).
