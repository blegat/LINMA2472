### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 32621224-a782-4bf6-9570-562cf2bb7360
using PlutoUI, DataFrames, PrettyTables, LinearAlgebra, Luxor, LaTeXStrings, MathTeXEngine, PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, PlutoTeachingTools

# ╔═╡ 6f72e8a5-819d-474c-a725-7f7318d964d7
include("utils.jl")

# ╔═╡ 5058e4eb-c53d-4468-b7ff-4f04ded96418
@htl("""
<p align=center style=\"font-size: 40px;\">Transformers in Large Language Models (LLMs)</p><p align=right><i>Benoît Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ 95ec4140-9147-11ef-2af4-5528bad0e6f5
md"# Large Language Models (LLMs)"

# ╔═╡ c09ec483-9fcf-48e7-b3c0-2508289e3cf3
md"## Autoregressive Models"

# ╔═╡ beccf4e8-1b01-4cb2-b23c-bc5db604f21c
md"""
Given a sequence of ``n_\text{ctx}`` past vectors ``x_{-1}, \ldots, x_{-n_\text{ctx}} \in \mathbb{R}^{n}``, "predict" the next ones. Key idea : *receding horizon*:

```math
\begin{align}
& p(x_0, x_1 | x_{-1}, \ldots, x_{-n_\text{ctx}})\\
& = p(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}})p(x_1 | x_0, x_{-1}, \ldots, x_{-n_\text{ctx}+1}, \textcolor{red}{x_{-n_\text{ctx}}})\\
& \approx p(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}}) p(x_{1} | x_0, x_{-1}, \ldots, x_{-n_\text{ctx}+1})
\end{align}
```

* **Model** : Probability of next vector ``\hat{p}(x_0 | X)`` where ``X`` concatenates ``x_{-1}, \ldots, x_{-n_\text{ctx}}``.
* **Loss** : Cross-entropy : ``\mathcal{L}_{\hat{p}}(X) \triangleq H(p, \hat{p}) = -\textbf{E}_p[\log(\hat{p})] = -\sum_{x_0} p(x_0 | X) \log(\hat{p}(x_0 | X))``
* Particular case for ``\hat{p}(x_0 | X) = \delta_y`` : ``\mathcal{L}_{\hat{p}}(X) = -\log(\hat{p}(y | X))``

#### What about Language Models ?

Given "past text", predict the "following text". How to turn text into vectors of ``\mathbb{R}^n`` ?
"""

# ╔═╡ ccf2dc71-b883-497a-bc58-29ffaf9ea4ad
md"## Text to vectors : step 1 → tokenization"

# ╔═╡ 1bbf2152-4fdf-4ed2-9bdf-95d699824d11
md"""
#### Why not encode each letter ?

* **Idea** : Turn each letter into its one-hot encoding in ``\mathbb{R}^{26}``.
* **Issue** : The "past text" only has ``n_\text{ctx}`` characters so ``n_\text{ctx}`` must be **large** but transformers have a complexity **quadratic** in ``n_\text{ctx}``!
* **Practical details** : Text is encoded with [UTF-8](https://en.wikipedia.org/wiki/UTF-8) so each character is encoded into 1 to 4 bytes. We encode each byte to a vector in ``\mathbb{R}^{256}`` but care must be taken not to generate invalid UTF-8.

#### Why not encode each word ?

* **Idea** : Turn each word into its one-hot encoding in ``\mathbb{R}^n``. The value of ``n`` is the number of words. Depending on the language ([source](https://en.wikipedia.org/wiki/List_of_dictionaries_by_number_of_words)):

| Language | French  | English |  Dutch  | German  |
|----------|---------|---------|---------|---------|
| ``n``    | 408,078 | 350,000 | 350,000 | 200,000 |

* **Issue** : The value of ``n`` is **too large**. We cannot trust the words of languages to be a tokenization that optimally compresses text for our dataset.
"""

# ╔═╡ 57c2c944-0d91-489d-8ad7-f5520e71ef3e
md"## Byte Pair Encoding"

# ╔═╡ c3db7eb2-356a-428f-9777-6369662d8b06
md"""
Note that the new tokens can also be part of the most frequence pair!
"""

# ╔═╡ 2e8b1a77-1f04-4035-8d82-4061d81ecb7a
md"## Increasing length of \"past text\""

# ╔═╡ ed5b5702-4cca-4116-a70f-4a562178f490
md"""
> **Challenging tradeoff**: Encode text to **increase** length of "past text" while keeping ``n_\text{ctx}`` and ``n`` **small** enough.

Length of "past text" increases with vocabulary size ``n_\text{voc}`` and context window ``n_\text{ctx}``.
"""

# ╔═╡ e2eca085-9f99-4e3a-9db4-e7f692aedd34
md"## Text to vectors : step 2 → embedding"

# ╔═╡ 9e898325-e9e2-45bd-af74-3dd86f00f7b5
md"""
Consider one-hot encoding with vocabulary size ``n_\text{voc}`` and a *bigram model*
```math
\hat{p}(x_0 | x_{-1}) = \text{softmax}(W_d \tanh(\cdots\tanh(W_1 x_{-1})\cdots)
```
The matrix ``W_d`` has ``n_\text{voc}`` rows and ``W_1`` has ``n_\text{voc}`` columns → issue if ``n_\text{voc}`` is large

**Embedding** : Use vectors ``c_1, \ldots, c_{n_\text{voc}} \in \mathbb{R}^{d_\text{emb}}`` with *embedding size* (aka *hidden size*) ``d_\text{emb} \ll n_\text{voc}``.

Equivalently, we still use one-hot encoding but we add an encoder
``C \in \mathbb{R}^{d_\text{emb} \times n_\text{voc}}`` and decoder ``D \in \mathbb{R}^{n_\text{voc} \times d_\text{emb}}``
```math
\hat{p}(x_0 | x_{-1}) = \text{softmax}(D W_d \tanh(\cdots\tanh(W_1 C x_{-1})\cdots)
```
"""

# ╔═╡ bcf7667f-f99b-4d10-af84-5d3879f1db5d
qa(
html"What difference do you expect with respect to the previous model ?",
md"""
The products ``W_1C`` and ``DW_d`` have the same dimension as the matrices ``W_1`` and ``W_d`` of the previous model. So the expressive power of the model was not improved while we increased the number of parameters and we potentially made the loss function "even more nonconvex".

If the hidden dimension (i.e., the number of rows of ``C`` / columns of ``W_1`` or the number of rows of ``W_d`` / columns of ``D``) is much smaller than ``n_\text{voc}``, then it's faster to compute ``W_1(Cx)``. Moreover, we are forcing the matrix ``W_1C`` to have a low rank compared the model without ``C``. This means less expressivness but it might also prevent overfitting so the case isn't so clear.

The case become clearer when the input embedding ``C`` is shared between more than one character, i.e., ``n_\text{ctx} > 1``.
Same for the output embedding, ``D`` is useful when it is not preceded by a linear with which it can just be merged.
""")

# ╔═╡ e86a57ae-3945-4cbf-b2e4-a96f4b5295e0
qa(md"Why don't we use ``D = C^{-1}`` ?",
md"""
The ``C`` matrix is not invertible as it is not square. Given a vector ``u``, the vector ``v = Du`` could be computed to be the mininum norm solution of ``u = Cv``, this is what is done by `v = D \ u`. Computing `D \ u` would be more computational work than `D' \ u` but that's not the only reason `D'` is used.
If ``v`` is chosen so that ``u = Cv``, it means that ``v`` is a linear combination allowing to reconstruct ``u`` using the columns of ``C``.
From that perspective, ``v_i`` could be nonzero even though the direction of the vector ``u`` us far from the direction of the column ``C_{:i}``.
That does not correspond to what we want to do here.

Here, we want a probability vector ``p`` with high probability ``p_i`` when the direction of the column ``C_{:i}`` is close to the direction of ``u``.
In the vector ``v = C^\top u``, ``v_i = \langle C_{:i}, u \rangle`` is the scalar product between the ``i``th column of ``C`` and ``u`` hence it is a good measure of how close the direction are to each other.
Of course this assumes that the column of ``C`` have the same norm, but the hope is for the model to figure in training that the columns of ``C`` should have unit norm.
We can then simply apply softmax to turn these scalar products into probabilities ``p = \text{softmax}(v)``.
""")

# ╔═╡ 6622f9f0-cecc-476e-9d49-7d651f433b9f
md"# Pre-transformers approaches"

# ╔═╡ 6aa690e9-389f-4398-abae-b95060db4d90
md"## Shared embedding"

# ╔═╡ 6712c883-b407-47e1-a666-4de05f8f8d6e
HAlign(
	md"""
```math
\begin{multline}
\hat{p}(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}}) = \\
	\text{softmax}(W_2 \tanh(W_1
\begin{bmatrix}
C x_{-1}\\
\vdots\\
C x_{-n_\text{ctx}}
\end{bmatrix}
))
\end{multline}
```
	""",
	img("bengio2000Neural", :width => 250),
)

# ╔═╡ c4bebd0d-eacf-4db4-b5b3-4dca50ab9e1b
qa(md"What are the number of columns of ``W_1`` and number of rows of ``W_2`` now ?",
   md"""
The matrix ``W_1`` has ``n_\text{ctx}d_\text{emb}`` columns. Assuming ``d_\text{emb} \ll n_\text{voc}`` and ``n_\text{ctx} \gg 1``, this is much smaller than the number ``n_\text{ctx}n_\text{voc}`` that we would have without the embedding. The number of rows of ``W_2`` is ``n_\text{voc}``, unaffected by the embedding.
""")

# ╔═╡ f8330700-e964-4e19-9c55-2b11df45789e
md"## Embedding sizes in LLMs"

# ╔═╡ 4e10271c-49f8-4f1d-869c-5fa11275d7f6
md"## Recurrent neural networks (RNN)"

# ╔═╡ d54b5390-0ec0-4ff8-ab18-51726482ca46
md"## Extensions of RNNs"

# ╔═╡ 6800afbf-8ac6-4308-b4cc-b37da57e42c1
md"# Attention is all you need"

# ╔═╡ 55435b26-7fc3-4c8b-8013-6fd4fb65a08e
md"## Numerical dictionary"

# ╔═╡ bcbb3db2-85b3-4cb0-9309-f5c032d14da5
md"
What would a numerical dictionary look like ? Consider keys ``k_i \in \mathbb{R}^{d_k}`` and values ``v_i \in \mathbb{R}^{d_v}``. Given a query ``q \in \mathbb{R}^{d_k}``,"

# ╔═╡ d558636d-c714-4033-ae73-5b92c3cdedf3
dict = Dict([1, 0] => [1, 1], [0, 1] => [-1, 1])

# ╔═╡ 70f395b2-f8c2-44d5-b0af-702659dd7fee
dict[[1, 0]]

# ╔═╡ b1a924f4-e2f0-445c-830f-94287a0e52f7
function numerical_lookup(dict, query)
	_, i = findmax([dot(query, key) for key in keys(dict)])
	return collect(values(dict))[i]
end

# ╔═╡ 95504a74-d5ef-4fb7-83a0-88914c7cbc59
numerical_lookup(dict, [0.8, 0.2])

# ╔═╡ 8d231f2c-4b0c-4c37-a746-16e98d4cafc8
md"## Attention head"

# ╔═╡ 570fa160-3adb-463e-99b8-b7dd05076908
function softmax(x)
	y = exp.(x)
	return y / sum(y)
end

# ╔═╡ 77f446ac-6030-48f2-9bea-93c427f9fcb9
function softmax_lookup(dict, query)
	ks = keys(dict)
	α = softmax([dot(query, key) for key in keys(dict)])
	@show α
	return sum(α * value for (α, value) in zip(α, values(dict)))
end

# ╔═╡ 1faa4ab2-6c93-47dc-b631-8be52780fe7d
softmax_lookup(dict, [0.8, 0.2])

# ╔═╡ bf563783-9784-4c74-a7b1-6d7a3ed618c5
md"## Matrix form of attention"

# ╔═╡ 9ff95a9a-192b-4a12-8e2e-7acd6659c066
md"""
```math
\begin{align}
Q & = \begin{bmatrix}
  q_1 & \cdots & q_{n_\text{ctx}}
\end{bmatrix} &
K & = \begin{bmatrix}
  k_1 & \cdots & k_{n_\text{ctx}}
\end{bmatrix} &
K^\top Q & =
\begin{bmatrix}
  \langle k_1, q_1 \rangle & \cdots & \langle k_1, q_{n_\text{ctx}} \rangle\\
  \vdots & \ddots & \vdots\\
  \langle k_{n_\text{ctx}}, q_1 \rangle & \cdots & \langle k_{n_\text{ctx}}, q_{n_\text{ctx}} \rangle
\end{bmatrix}
\end{align}
```
"""

# ╔═╡ 76ba4e9b-8bb0-47c4-b607-2ca711f035e6
md"## Masked Attention"

# ╔═╡ 8c27b182-0c3c-4c19-9619-df62b7dd6bf0
HAlign(
md"""
💡 **Key idea** In the model for ``\hat{p}(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}})``, incorporate sub-models
```math
\begin{align}
\bar{p}(&x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}})\\
\bar{p}(&x_{-1} | x_{-2}, \ldots, x_{-n_\text{ctx}})\\
& \quad\qquad\vdots\\
\bar{p}(&x_{-n_\text{ctx}+1} | x_{-n_\text{ctx}}).
\end{align}
```
""",
md"""
Mask prevent ``\hat{p}`` to look input the future:
```math
M
=
\begin{bmatrix}
  0 & 0 & \cdots & 0\\
  -\infty & 0 & \ddots & \vdots\\
  \vdots & \ddots & \ddots & 0\\
  -\infty & \cdots & -\infty & 0
\end{bmatrix}
```
""",
)

# ╔═╡ 0c0c1163-0aec-4089-9acc-539b3a86d0b3
md"""
```math
\text{Masked-Attention}(V, K, Q)\
=
V\text{softmax}(M + K^\top Q/\sqrt{d_k})
```
"""

# ╔═╡ b7583418-f4fb-4c63-b421-b5b9af269768
md"## Multi-Head Attention"

# ╔═╡ 6fc13413-53de-4c75-9b9e-620e0b7f8a1f
qa(md"Is ``W^O`` needed if ``h = 1`` ?", md"No, if ``h = 1``, we can merge ``W^OW_1^V`` into a new ``W_1^V``.")

# ╔═╡ d05e6f0f-0081-4fb6-91e9-ac2f58beda4a
md"# Decoder-only transformer"

# ╔═╡ a3efd921-eb14-4901-9d6c-800cc812fe02
md"## Self-Attention"

# ╔═╡ 4b61363d-87c9-4755-8286-44df34e9dd6a
qa(
html"Is the order between the tokens taken into account by the model ?",
md"""
No. Since the same matrices ``W_j^V``, ``W_j^K`` and ``W_j^Q`` multiply the different position. The **position** information is completely **lost**!
"""
)

# ╔═╡ 453544fc-0e3e-4e04-8c0c-192f3a038884
md"## Positional encoding"

# ╔═╡ 92e01e21-ca77-43fc-9bf8-0c5a7aaed1bb
md"## Residual connection"

# ╔═╡ f2cba2aa-c541-4692-a441-e65741750a15
md"## Layer normalization"

# ╔═╡ e383bb72-49a1-4df1-84c3-b95a2ffe00f5
md"## Feed-Forward network"

# ╔═╡ af8194a1-a358-4cf7-b446-6b377cb76687
md"The feed-forward network is implemented **independently** for the output of each query so each query can be processed independently through each **layer**. The next layer allows each queries to then look at the results of the previous layer for **past** (because of the mask) queries."

# ╔═╡ 79e6c4a8-cc1e-40cc-bb09-e9a7a9a8e475
md"## Transformer variations"

# ╔═╡ a5b20939-9afa-48c0-aa67-cbca6bc99804
md"## Cost of LLMs"

# ╔═╡ a14e505e-2e4a-4c73-8133-7560ba58916b
md"## Key-Value (KV) cache"

# ╔═╡ 9a8eef1b-27c0-4d57-a389-53708ade9058
md"""
Let ``\hat{Y}_{i}`` be the intermediate output of ``i \in \{1, \ldots, N\}``.
The the columns of the matrix ``\text{softmax}(C^\top \hat{Y}_i)`` (column-wise softmax) can be thought as intermediate probabilities that we denote ``\hat{p}_i``:
```math
(\hat{p}_i(x_{-n_\text{ctx}+1} | x_{-n_\text{ctx}}), \ldots, \hat{p}_i(x_{-1} | x_{-2}, \ldots, x_{-n_\text{ctx}}), \hat{p}_i(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}}))
```
and we predict the next token using ``\hat{p}_N(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}})``.
"""

# ╔═╡ 728f5fdf-77a5-46c7-b3ee-01064ef1b7e2
qa(md"Should we discard all these intermediate ``\hat{Y}_i`` we computated or can we reuse it for the following token ?",
md"""
For the next token, the corresponding intermediate probabilities would be:
```math
(\hat{p}_i(x_{-n_\text{ctx}+2} | x_{-n_\text{ctx}+1}), \ldots, \hat{p}_i(x_{0} | x_{-1}, \ldots, x_{-n_\text{ctx}+1}), \hat{p}_i(x_1 | x_{0}, \ldots, x_{-n_\text{ctx}+1}))
```
Note that
```math
\begin{align}
   \hat{p}_i(x_{-n_\text{ctx}+2} | x_{-n_\text{ctx}+1})
   & \approx
   \hat{p}_i(x_{-n_\text{ctx}+2} | x_{-n_\text{ctx}+1}, x_{-n_\text{ctx}})\\
   \hat{p}_i(x_{0} | x_{-1}, \ldots, x_{-n_\text{ctx}+1})
   & \approx
   \hat{p}_i(x_0 | x_{-1}, \ldots, x_{-n_\text{ctx}})
\end{align}
```
So for any ``j < n_\text{ctx}``, the ``j``th column of the ``\hat{Y}_i'`` that should be computed for the new token is approximately equal to
the ``(j+1)``th column of ``\hat{Y}_i`` that we already computed for the previous token.
What's more, the column of ``\hat{Y}_i`` was computed with one more token as context compared to what we need to compute in ``\hat{Y}_i'``.
So even though it's not equal, reusing what we computed in ``\hat{Y}_i`` should provide better result, assuming the trained transformers using this auto-regressive structure in his layers.
""")

# ╔═╡ c8923675-e73e-4621-82b9-966d8b003b97
md"# Encoder-decoder transformer"

# ╔═╡ 04e9b912-6712-4290-acc4-f24bb27a1469
md"## Machine translation"

# ╔═╡ 8b78360a-21cb-4574-a84d-46ea4d0cedb1
img("sutskever2014Sequence")

# ╔═╡ 6bff7bca-ea1d-44c6-b8c3-040250f90654
md"## Cross-Attention"

# ╔═╡ f572e113-b36b-4a6b-96c7-c26f100e1ad4
md"## Utils"

# ╔═╡ f6f7376e-9984-4289-b8ff-9d47e5358791
import DocumenterCitations, CSV, Logging

# ╔═╡ 1d5b1b7c-828c-4a16-b446-cff21b015d45
biblio = load_biblio!()

# ╔═╡ 94ae440d-0644-49db-9461-f1a1ff1d7f87
cite(args...) = bibcite(biblio, args...)

# ╔═╡ f4366cf6-2be0-42b8-96c4-120be3f5c25e
md"""
### References

* Recurrent neural networks : $(cite("goodfellow2016Deep", "Chapter 10")) and Section 4.7 of [The Elements of Differentiable Programming book](https://diffprog.github.io/)
* Transformers : $(cite("vaswani2017Attentiona")) and Section 4.8 of [The Elements of Differentiable Programming book](https://diffprog.github.io/)
* [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) by Andrej Karpathy
"""

# ╔═╡ 0583ee0c-3802-4e81-b179-a80a82493b43
md"""
Byte Pair Encoding algorithm $(cite("sennrich2016Neural")) greedily merges the most frequent pair of tokens over the dataset into a new token.
Most used implementations are `SentencePiece` $(cite("kudo2018SentencePiece")) and `tiktoken` (play with it [here](https://tiktokenizer.vercel.app/)). For instance, on [this example](https://en.wikipedia.org/wiki/Byte_pair_encoding), the pair `('a', 'a')` is the most frequent so we substitute it by a new token, say `'Z'`:
"""

# ╔═╡ 2a7e5096-1e8d-4506-96d2-86de0a7d39aa
md"""
Forcing ``D = C^\top`` appears to work well in practice $(cite("press2017Using")), this is what is used in $(cite("vaswani2017Attentiona")).
"""

# ╔═╡ 9cb90e76-3bb5-41ff-bc79-c4949400d904
md"""
With ``n_\text{ctx} > 1``, the encoder ``C`` is shared by all tokens:
See for instance the network below taken from $(cite("bengio2000Neural", "Figure 1")), the first popular application of neural nets for languages:
"""

# ╔═╡ 55a09acc-84da-491c-86ba-9a66f4ea52fe
HAlign(
md"""
```math
\begin{align}
h^{(t+1)} & = \tanh(Wh^{(t)} + Ux^{(t+1)} + b)\\
o^{(t)} &= Vh^{(t)} + c\\
\hat{y}^{(t)} &= \text{softmax}(o^{(t)})
\end{align}
```
Illustrated on the right $(cite("goodfellow2016Deep", "Figure 10.3")).

RNNs as language model showcased in $(cite("mikolov2010Recurrent")).

**Issue**: Training time and space complexity is proportional to ``n_\text{ctx}`` and **cannot parallelize** to speed up.
""",
img("RNN")
)

# ╔═╡ 225e58ba-b78d-4a0a-be4f-ad642c879b93
md"""
It's difficult to model long-term dependencies as their gradient either vanish or explodes exponentially (think of the power method) $(cite("goodfellow2016Deep", "Section 10.7"))

*Gated* extensions attempting to solve this issue $(cite("goodfellow2016Deep", "Section 10.10")):

* Long short-term memory (LSTM) $(cite("graves2014Generating"))
* Gated recurrent unit (GRU) $(cite("cho2014Properties"))
"""

# ╔═╡ a21fbc70-9137-4d0e-8c8c-cbdc5269778f
md"""
Recently, Mamba suggests a solution to the complexity issue $(cite("gu2024Mamba")). As it scales better with ``n_\text{ctx}``, it is even suggested to get rid of the tokenizer : $(cite("wang2024MambaByte")).
"""

# ╔═╡ 86101f07-67c5-4df2-911c-4013c44d6c5b
md"""
Attention head provides a differentiable numerical dictionary $(cite("bahdanau2016Neural"))
```math
\begin{align}
\alpha
& =
\text{softmax}(\langle q, k_1 \rangle, \ldots, \langle q, k_{n_\text{ctx}}\rangle)
&
\text{Attention}(q, k, v)
& =
\sum_{i=1}^{n_\text{ctx}} \alpha_i v_i
\end{align}
```
"""

# ╔═╡ 5150d8f3-6e85-43f2-801a-eae5cc3e3095
HAlign(
	md"""
`softmax` is then applied to each **column**:
```math
\text{softmax}(K^\top Q/\sqrt{d_k})
```
Division by ``\sqrt{d_k}`` scales the input of softmax to
preferable regions $(cite("vaswani2017Attentiona", "Secton 3.2.1")).

Illustrated on the right from $(cite("bahdanau2016Neural", "Figure 3(a)")).

```math
\text{Attention}(V, K, Q) = V\text{softmax}(K^\top Q/\sqrt{d_k})
```
""",
	img("attention_matrix"),
)

# ╔═╡ d014e6aa-92f6-4ca1-be47-516565d1bb20
HAlign((
	md"""
Heads focus on different aspects. Their outputs are **combined** with ``W^O \in \mathbb{R}^{d_\text{emb} \times hd_v}``:
```math
\begin{align}
	\text{head}_j & = \text{Attention}(W_j^VV, W_j^KK, W_j^QQ)\\
	\text{MultiHead}(V, K, Q)
	& =
	W^O\text{vcat}(\text{head}_1, \ldots, \text{head}_h)
\end{align}
```
See $(cite("vaswani2017Attentiona", "Figure 2")) on the right.

Similarly, in the masked case:
```math
\begin{align}
	\text{head}_j & = \text{Masked-Attention}(W_j^VV, W_j^KK, W_j^QQ)\\
	\text{Masked-MultiHead}&(V, K, Q)
	=
	W^O\text{vcat}(\text{head}_1, \ldots, \text{head}_h)
\end{align}
```
""",
	img("multi-head", :width => 250)),
	[70, 30],
)

# ╔═╡ 25b79953-fd7c-46c1-b760-d57c09910981
qa(md"""
How does the number of parameters of transformers compare with $(cite("bengio2000Neural")) or RNNs for large ``n_\text{ctx}`` ?
""",
md"""
* The number of parameters of the transformer does **not** depend on ``n_\text{ctx}``.
* The number of parameters of $(cite("bengio2000Neural")) depends linearly with ``n_\text{ctx}``. Assuming that the number of hidden neurons scales proportionally with ``n_\text{ctx}``, the number of parameters even scales quadratically with ``n_\text{ctx}``!
* For RNNs, if the dimension of the internal state scales proportionally with ``n_\text{ctx}``, the number of parameters is also proportional with ``n_\text{ctx}``! If the dimension of the internal state is kept too small, increasing the context won't be so helpful, due to *encoder bottleneck*, see next slide.
""")

# ╔═╡ c1437dcc-22cb-424f-9b8e-326172f82d86
md"""
* LSTM **encoder** → **context** → LSTM **decoder** $(cite("sutskever2014Sequence")). See $(cite("sutskever2014Sequence", "Figure 1")) below.
* Issue with *encoder bottleneck*. All information has to be summarized in the **context**.
"""

# ╔═╡ 4f1d5112-dbac-4eb6-8518-0dc4193c3f8e
bib(args...) = bibrefs(biblio, args...)

# ╔═╡ 61dc1905-338f-4bfd-a158-2f6bacff769e
bib(["goodfellow2016Deep", "vaswani2017Attentiona"])

# ╔═╡ 4df0a18d-cb14-41b1-ba40-fd6bfcbb0b03
bib(["sennrich2016Neural", "kudo2018SentencePiece"])

# ╔═╡ 97463c54-7cc7-4497-a8a6-6422f5f582bd
bib(["team2024Gemini", "team2024Geminia", "team2024Gemma", "team2024Gemmaa", "sennrich2016Neural", "radford2019Language", "brown2020Language", "touvron2023Llama", "yu2023MEGABYTE"])

# ╔═╡ eb18303f-3dfb-4b87-90f2-f6dc542d7221
bib(["press2017Using", "vaswani2017Attentiona"])

# ╔═╡ 76e2f97b-1c06-40cd-b134-d5155aa5587d
bib(["bengio2000Neural"])

# ╔═╡ 75ca478c-916f-464a-9435-8208ee726d50
bib(["team2024Gemma", "team2024Gemmaa", "radford2019Language", "touvron2023Llama"])

# ╔═╡ 5b4a67a9-e33e-4dc6-b9f0-fd9a2cca6f2a
bib(["mikolov2010Recurrent", "goodfellow2016Deep"])

# ╔═╡ 8eafcfed-9771-4d99-b0c5-bd75a6dab012
bib(["cho2014Properties", "graves2014Generating", "goodfellow2016Deep", "gu2024Mamba", "wang2024MambaByte"])

# ╔═╡ e41d13ca-1dc1-45ae-9fa6-a83c4101120d
bib(["bahdanau2016Neural"])

# ╔═╡ c032b3ff-c539-4e38-81d0-39b28b3a8076
bib(["bahdanau2016Neural", "vaswani2017Attentiona"])

# ╔═╡ b56e9e56-e74a-401b-b4b5-f36bb33341d5
bib(["he2015Deep"])

# ╔═╡ 2a8433e3-9a3b-487b-abf3-09278ea42389
bib(["ioffe2015Batch", "ba2016Layer", "vaswani2017Attentiona"])

# ╔═╡ 4dd7083a-e730-4f4b-bde8-fc1a5b08ebfc
bib(["he2016Identity", "radford2019Language", "su2023RoFormer"])

# ╔═╡ 45efc71d-d5f8-474e-9b89-e72fac7110fd
bib("bengio2000Neural")

# ╔═╡ f7ca738d-5215-4e91-a2f3-a5ff10911313
bib("sutskever2014Sequence")

# ╔═╡ 85a10748-8d19-44a8-a1c5-0d13b093f1bf
function draw_transformer(decoder_only = true)
	scale(0.4, 0.4)
	Luxor.placeimage(readpng("images/transformer.png"), centered = true)
	if decoder_only
		sethue("red")
		setopacity(0.4)
		box(Point(-350, -160), Point(320, 20), :fill)
		box(Point(-350, 20), Point(0, 460), :fill)
		translate(Point(-170, -190))
		setopacity(1)
		fontsize(32)
		text("Not used for now", halign = :center)
	end
end

# ╔═╡ d1ba8da3-add8-4dbe-9ebf-9a32fa5cd5dd
HAlign(
md"""
*Pre-activation* for residual neural networks introduced in $(cite("he2016Identity")) and used in GPT-2 $(cite("radford2019Language")). See figure on the right.

*Rotary Positional Encoding* $(cite("su2023RoFormer")) replaces
``W^K(Cx_i + p_i)`` and ``W^Q(Cx_i + p_i)``
by ``R^i W^KCx_i`` and ``R^i W^QCx_i`` where ``R`` is a rotation matrix.
Advantage : ``\langle k_i, q_j \rangle`` contains ``R^{i - j}`` → **relative** difference of position.
""",
HTML(html(@draw begin
	draw_transformer()
	sethue("blue")
	scale(2, 2)
	arrow(Point(175, -36), Point(180, -43), Point(170, -48), Point(145, -53), :stroke, startarrow=false, finisharrow=true)
	arrow(Point(175, -30), Point(190, -10), Point(195, 10), Point(145, 17), :stroke, startarrow=false, finisharrow=true)
	arrow(Point(175, 120), Point(190, 130), Point(195, 150), Point(145, 192), :stroke, startarrow=false, finisharrow=true)
end 300 400)),
)

# ╔═╡ 8d6ec2b3-997e-4df5-a3b2-c1dffa53d0ec
qa(
	md"What is the time complexity of inference with respect to ``d_\text{emb}``, ``n_\text{voc}``, ``n_\text{ctx}``, ``d_\text{ff}``, ``h`` and ``N`` ?",
HAlign(
md"""
| Input | Parameters | Time |
|-------|------------|------|
| ``CX + P \in \mathbb{R}^{d_\text{emb} \times n_\text{ctx}}`` | ``W_j^V \in \mathbb{R}^{d_v \times d_\text{emb}}`` | ``O(d_v d_\text{emb} n_\text{ctx})`` |
| ``CX + P \in \mathbb{R}^{d_\text{emb} \times n_\text{ctx}}`` | ``W_j^K, W_j^Q \in \mathbb{R}^{d_k \times d_\text{emb}}`` | ``O(d_k d_\text{emb} n_\text{ctx})`` |
| ``K, Q \in \mathbb{R}^{d_k \times n_\text{ctx}}`` |  | ``O(d_k n_\text{ctx}^2)`` |
| ``V \in \mathbb{R}^{d_v \times n_\text{ctx}}, ... \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}`` |  | ``O(d_v n_\text{ctx}^2)`` |
| ``... \in \mathbb{R}^{d_v \times n_\text{ctx}}`` | ``W^O \in \mathbb{R}^{d_\text{emb} \times d_v}`` | ``O(d_\text{emb} d_v n_\text{ctx})`` |
| ``... \in \mathbb{R}^{d_\text{emb} \times n_\text{ctx}}`` | ``W_1 \in \mathbb{R}^{d_\text{ff} \times d_\text{emb}}`` | ``O(d_\text{emb} d_\text{ff} n_\text{ctx})`` |
| ``... \in \mathbb{R}^{d_\text{ff} \times n_\text{ctx}}`` | ``W_2 \in \mathbb{R}^{d_\text{emb} \times d_\text{ff}}`` | ``O(d_\text{emb} d_\text{ff} n_\text{ctx})`` |

So for ``N`` layers (ignoring the complexity of the embedding):
```math
O(Nn_\text{ctx}(n_\text{ctx}(d_v + d_k) + d_\text{emb}(d_v+d_k+d_\text{ff})))
```
Assuming that ``d_v, d_k, d_\text{ff}`` has the same scale as ``d_\text{emb}``:
```math
O(Nn_\text{ctx}^2d_\text{emb} + Nn_\text{ctx}d_\text{emb}^2)
```
""",
HTML(html(@draw begin
	draw_transformer()
	translate(-10, 150)
	scale(0.6)
	Luxor.placeimage(readpng("images/multi-head.png"), centered = true)
end 300 400))
)
)

# ╔═╡ a873f760-bfc1-489f-a58e-75e12afa54f2
function highlight(a, b, c, d)
	sethue("green")
	setopacity(0.4)
	#box(Point(a, b), Point(c, d), :fill)
	polysmooth(box(Point(a, b), Point(c, d), vertices=true), 10, action = :fill)
	setopacity(1)
	polysmooth(box(Point(a, b), Point(c, d), vertices=true), 10, action = :stroke)
end

# ╔═╡ b9caae1a-38aa-4d01-9cda-3d6782fb0e03
HAlign(md"""
*Self-Attention* with embedding ``C`` is:
```math
\text{Masked-MultiHead}(CX, CX, CX)
```

The embedding vectors ``CX`` take then different projections
for value, key, query and also for different heads!
```math
\text{head}_j = \text{Masked-Attention}(W_j^VCX, W_j^KCX, W_j^QCX)
```
""",
HTML(html(@draw begin
	draw_transformer()
	highlight(200, 250, 375, 350)
end 300 400)),
)

# ╔═╡ c5be3956-5102-4d88-bfdb-9813c0555fe1
HAlign(
md"""
Cannot sum ``Cx_i + e_i`` with one-hot encoding ``e_i \in \mathbb{R}^{n_\text{ctx}}`` as the dimension of ``Cx_i`` is ``\mathbb{R}^{d_\text{emb}}``.

So we also add a positional embedding ``P`` : ``Cx_i + Pe_i = Cx_i + p_i``.

With Self-Attention:
```math
\text{Self-MultiHead}(CX + P, CX + P, CX + P)
```
""",
HTML(html(@draw begin
	draw_transformer()
	highlight(310, 410, 530, 510)
end 300 400))
)

# ╔═╡ 18c26901-85eb-45ac-89bf-b03bd255007a
HAlign(
md"""
Residual connection $(cite("he2015Deep"))
$(img("resnet"))
""",
HTML(html(@draw begin
	draw_transformer()
	highlight(210, -85, 290, -45)
	highlight(360, -75, 440, 50)
	highlight(210, 210, 290, 250)
	highlight(360, 220, 440, 420)
end 300 400))
)

# ╔═╡ 5f05e717-a51a-4a99-bb11-cc493217f93f
HAlign(
md"""
Norm of gradient increases exponentially with depth.
Issue for deep neural net.
Consider output
```math
\begin{bmatrix}
  y_{1,1} & \ldots & y_{1,d_\text{emb}}\\
  \vdots & \ddots & \vdots\\
  y_{d_\text{batch},1} & \ldots & y_{d_\text{batch},d_\text{emb}}
\end{bmatrix}
```
Normalization : ``y_{i,j} \mapsto g(y_{i,j} - \mu_{i,j})/\sigma_{i,j}`` for gain ``g``, mean ``\mu`` and standard deviation ``\sigma``.

* Batch normalization : ``\sigma_{i,j} = \sigma_{j}`` $(cite("ioffe2015Batch"))
* Layer normalization : ``\sigma_{i,j} = \sigma_{i}`` $(cite("ba2016Layer"))

Batch norm depends on the batch hence [is tricky to implement](https://www.youtube.com/watch?v=P6sfmUTpUmc). Layer normalization is used in $(cite("vaswani2017Attentiona")).
""",
HTML(html(@draw begin
	draw_transformer()
	highlight(290, -85, 360, -45)
	highlight(290, 210, 360, 250)
end 300 400))
)

# ╔═╡ 3d8add97-59e1-444a-838b-85c2a2ac60b3
HAlign(
md"""
*Cross-Attention* between
* values and keys ``E(CX + P)`` where ``E`` is the encoder, and ``X`` is the matrix of input tokens
* query ``Q`` depending on past output ``Y`` and number of layers already applied
```math
\text{MultiHead}(E(CX + P), E(CX + P), Q)
```

The embedding vectors ``CX`` take then different projections
for value, key, query and also for different heads!
```math
\begin{multline}
\text{head}_j = \text{Attention}(W_j^VV, W_j^KK, W_j^QQ)\\
\text{where } V = K = E(CX + P)
\end{multline}
```
""",
HTML(html(@draw begin
	draw_transformer(false)
	highlight(31, -97, 205, 5)
	#sethue("red")
	#setopacity(1)
	fontsize(32)
	text("CX + P", Point(-60, 252), halign = :center)
	text("CY + P", Point(65, 252), halign = :center)
	text(L"E(CX + P)", Point(-70, -160), halign = :center)
end 300 400)),
)

# ╔═╡ d050a7ee-3aa7-4539-a236-5b6446599ded
struct BPE
	text::String
	pairs::Dict{Tuple{Char,Char},Char}
end

# ╔═╡ 29474a70-32eb-4281-8626-87819afa7267
function add_pair(bpe::BPE, subs)
	pairs = copy(bpe.pairs)
	push!(pairs, subs)
	return BPE(replace(bpe.text, prod(subs.first) => subs.second), pairs)
end

# ╔═╡ 89305cae-098f-4644-9109-d00f1e3bc04c
function pair_stats(text::String)
	stats = Dict{Tuple{Char,Char},Int}()
	for i in eachindex(text)
		j = nextind(text, i)
		if j > lastindex(text)
			break
		end
		a = text[i]
		b = text[j]
		stats[(a, b)] = get(stats, (a, b), 0) + 1
	end
	return stats
end

# ╔═╡ 728c16b7-50cd-43fe-a0d7-61d37952a6b3
pair_stats("aaabdaaabac")

# ╔═╡ c7f318b9-30e6-4b79-b7da-52f70904d246
function substitute(text::String, pair::Tuple{Char,Char})
	new_char = min('Z' + 1, minimum(text)) - 1
	return replace(text, prod(pair.first) => pair.second)
end

# ╔═╡ f1afaf8c-d9ad-446a-9826-9c4cda19993f
new_token(text::String) = new_token(BPE(text, Dict()))

# ╔═╡ 01372b00-ecb2-42bd-b408-13234717d969
function new_token(bpe::BPE)
	stats = pair_stats(bpe.text)
	pair = findmax(stats)[2]
	new_char = min('Z' + 1, minimum(bpe.text)) - 1
	return add_pair(bpe, pair => new_char)
end

# ╔═╡ f7ca3ff7-b5cf-452b-b955-7219e7397324
iter_1 = new_token("aaabdaaabac")

# ╔═╡ 736920df-e4bb-4535-b982-e397aa0a782d
iter_2 = new_token(iter_1)

# ╔═╡ c3a9a0ce-3450-4b17-8696-2ab8534b29f2
iter_3 = new_token(iter_2)

# ╔═╡ 579a203b-e6f7-4190-b874-18b00a5c3f77
function load_llms()
	llms = DataFrame(CSV.File("llms.csv"))
	rename!(llms, "Embedding dimension" => "``d_\\text{emb}``")
	rename!(llms, "Vocabulary size" => "``n_\\text{voc}``")
	rename!(llms, "Context window" => "``n_\\text{ctx}``")
	rename!(llms, "Feed-Forward hidden dimension" => "``d_\\text{ff}``")
	return llms
end

# ╔═╡ f39305ea-f7f5-440e-ac55-c83e27f6e7fc
llms = load_llms()

# ╔═╡ 93200f46-7c8f-4362-a445-43c57b50a2d2
names(llms)

# ╔═╡ 7e27c349-ee76-46bd-b1c2-a9ce54974e10
function table(df; mandatory_columns = String[], included_columns = nothing)
	for col in mandatory_columns
		df = df[(!ismissing).(df[!, col]), :]
	end
	if !isnothing(included_columns)
		df = unique(df[!, included_columns])
	end
	Markdown.parse(pretty_table(
		String,
		sort(df),
		backend = :markdown,
		column_labels = names(df),
		allow_markdown_in_cells = true,
		formatters = [(v, _, _) -> ismissing(v) ? "" : v],
	))
end

# ╔═╡ 771d39a5-74dc-494e-929e-1164bb08b983
table(llms, mandatory_columns = ["``n_\\text{ctx}``"], included_columns = [
	"Name",
	"Ref",
	"``n_\\text{voc}``",
	"``n_\\text{ctx}``",
	"Tokenizer",
])

# ╔═╡ 91abc03b-fef7-4f93-96fc-13f1cf654f0d
HAlign(
md"""
See the table below for the size of embeddings of large language models:
$(table(llms, mandatory_columns = ["``d_\\text{emb}``"], included_columns = [
	"Name",
	"Num params",
	"Ref",
	"``n_\\text{voc}``",
	"``d_\\text{emb}``",
]))
""",
HTML(html(@draw begin
	draw_transformer()
	highlight(200, -160, 375, -110)
	highlight(200, 490, 375, 565)
end 300 400));
)

# ╔═╡ f95a6de6-5e02-4237-88ba-ec44ef3d38c3
HAlign(
md"""
Different weights
``W_1 \in \mathbb{R}^{d_\text{ff} \times d_\text{emb}}``, ``W_2 \in \mathbb{R}^{d_\text{emb} \times d_\text{ff}}`` for each layer:
```math
x \mapsto W_2\max(0, W_1x + b_1) + b_2
```
Expansion factor ``d_\text{ff} / d_\text{emb}`` is typically 4× like suggested in $(cite("vaswani2017Attentiona")) (but not for Gemma)
$(table(llms, mandatory_columns = ["``d_\\text{ff}``"], included_columns = [
	"Name",
	"Ref",
	"``d_\\text{emb}``",
    "``d_\\text{ff}``",
]))
""",
HTML(html(@draw begin
	draw_transformer()
	highlight(200, -45, 375, 25)
end 300 400)),
)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DocumenterCitations = "daee34ce-89f3-4625-b898-19384cb65244"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Luxor = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.8.1"
DocumenterCitations = "~1.4.1"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Luxor = "~4.3.0"
MathTeXEngine = "~0.6.6"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.72"
PrettyTables = "~3.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "8cf5017c347c067a952c4fa0584b3d429744517c"

[[deps.ANSIColoredPrinters]]
git-tree-sha1 = "574baf8110975760d391c710b6341da1afa48d8c"
uuid = "a4c015fc-c6ff-483c-b24f-f7ea428134e9"
version = "0.0.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.BibInternal]]
deps = ["TestItems"]
git-tree-sha1 = "b3107800faf461eca3281f89f8d768f4b3e99969"
uuid = "2027ae74-3657-4b95-ae00-e2f7d55c3e64"
version = "0.3.7"

[[deps.BibParser]]
deps = ["BibInternal", "DataStructures", "Dates", "JSONSchema", "TestItems", "YAML"]
git-tree-sha1 = "33478bed83bd124ea8ecd9161b3918fb4c70e529"
uuid = "13533e5b-e1c2-4e57-8cef-cac5e52f6474"
version = "0.2.2"

[[deps.Bibliography]]
deps = ["BibInternal", "BibParser", "DataStructures", "Dates", "FileIO", "TestItems", "YAML"]
git-tree-sha1 = "0f25be9708ae20d7b94d3bf9d0a91defcca4c884"
uuid = "f1be7e48-bf82-45af-a471-ae754a193061"
version = "0.3.0"

[[deps.Bijections]]
git-tree-sha1 = "a2d308fcd4c2fb90e943cf9cd2fbfa9c32b69733"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Documenter]]
deps = ["ANSIColoredPrinters", "AbstractTrees", "Base64", "CodecZlib", "Dates", "DocStringExtensions", "Downloads", "Git", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "MarkdownAST", "Pkg", "PrecompileTools", "REPL", "RegistryInstances", "SHA", "TOML", "Test", "Unicode"]
git-tree-sha1 = "47ffb8f27ffc01e2e57e7ae5365ae5ceef87b03d"
uuid = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
version = "1.14.1"

[[deps.DocumenterCitations]]
deps = ["AbstractTrees", "Bibliography", "Bijections", "Dates", "Documenter", "Logging", "Markdown", "MarkdownAST", "OrderedCollections", "Unicode"]
git-tree-sha1 = "c9953a03a0049333bec89ac254ea28e86fa7a1a9"
uuid = "daee34ce-89f3-4625-b898-19384cb65244"
version = "1.4.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Git]]
deps = ["Git_LFS_jll", "Git_jll", "JLLWrappers", "OpenSSH_jll"]
git-tree-sha1 = "824a1890086880696fc908fe12a17bcf61738bd8"
uuid = "d7ba0133-e1db-5d97-8f8c-041e4b3a1eb2"
version = "1.5.0"

[[deps.Git_LFS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bb8471f313ed941f299aa53d32a94ab3bee08844"
uuid = "020c3dae-16b3-5ae5-87b3-4cb189e250b2"
version = "3.7.0+0"

[[deps.Git_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "Libiconv_jll", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e2aef26f7d273f1e5b1daba56837c47b49b4388f"
uuid = "f8c6e375-362e-5223-8a59-34ff63f689eb"
version = "2.51.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSONSchema]]
deps = ["Downloads", "JSON", "URIs"]
git-tree-sha1 = "d13f79c4242969874da7d00bda17d59bc7699aa7"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "1.5.0"

    [deps.JSONSchema.extensions]
    JSONSchemaJSON3Ext = "JSON3"

    [deps.JSONSchema.weakdeps]
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "277779adfedf4a30d66b64edc75dc6bb6d52a16e"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.6"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "0f2da712350b020bc3957f269c9caad516383ee0"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "Cairo_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pango_jll", "XML2_jll", "gdk_pixbuf_jll"]
git-tree-sha1 = "e6ab5dda9916d7041356371c53cdc00b39841c31"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.54.7+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "e24491cb83551e44a69b9106c50666dea9d953ab"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.4"

[[deps.Luxor]]
deps = ["Base64", "Cairo", "Colors", "DataStructures", "Dates", "FFMPEG", "FileIO", "PolygonAlgorithms", "PrecompileTools", "Random", "Rsvg"]
git-tree-sha1 = "54bdbc3b05b3a4cf25ec4c00054038758c1c090b"
uuid = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
version = "4.3.0"

    [deps.Luxor.extensions]
    LuxorExtLatex = ["LaTeXStrings", "MathTeXEngine"]
    LuxorExtTypstry = ["Typstry"]

    [deps.Luxor.weakdeps]
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MarkdownAST]]
deps = ["AbstractTrees", "Markdown"]
git-tree-sha1 = "465a70f0fc7d443a00dcdc3267a497397b8a3899"
uuid = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"
version = "0.1.2"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "a370fef694c109e1950836176ed0d5eabbb65479"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.6"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenSSL_jll", "Zlib_jll"]
git-tree-sha1 = "301412a644646fdc0ad67d0a87487466b491e53d"
uuid = "9bd350c2-7e96-507f-8002-3f2e150b4e1b"
version = "10.2.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "f53232a27a8c1c836d3998ae1e17d898d4df2a46"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.72"

[[deps.PolygonAlgorithms]]
git-tree-sha1 = "809227325f22eedaf6f9eaac311247950678ec8d"
uuid = "32a0d02f-32d9-4438-b5ed-3a2932b48f96"
version = "0.3.3"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6b8e2f0bae3f678811678065c09571c1619da219"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "2d155af8d27cc03e39771aac4468695bcedb6ca7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.10.0"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "e53dad0507631c0b8d5d946d93458cbabd0f05d7"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.1.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TestItems]]
git-tree-sha1 = "42fd9023fef18b9b78c8343a4e2f3813ffbcefcb"
uuid = "1c621080-faea-4a02-84b6-bbd5e436b8fe"
version = "1.0.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.YAML]]
deps = ["Base64", "Dates", "Printf", "StringEncodings"]
git-tree-sha1 = "2f58ac39f64b41fb812340347525be3b590cce3b"
uuid = "ddb6d928-2868-570f-bddf-ab3f9cf99eb6"
version = "0.4.14"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "895f21b699121d1a57ecac57e65a852caf569254"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.13+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─5058e4eb-c53d-4468-b7ff-4f04ded96418
# ╟─95ec4140-9147-11ef-2af4-5528bad0e6f5
# ╟─f4366cf6-2be0-42b8-96c4-120be3f5c25e
# ╟─61dc1905-338f-4bfd-a158-2f6bacff769e
# ╟─c09ec483-9fcf-48e7-b3c0-2508289e3cf3
# ╟─beccf4e8-1b01-4cb2-b23c-bc5db604f21c
# ╟─ccf2dc71-b883-497a-bc58-29ffaf9ea4ad
# ╟─1bbf2152-4fdf-4ed2-9bdf-95d699824d11
# ╟─57c2c944-0d91-489d-8ad7-f5520e71ef3e
# ╟─0583ee0c-3802-4e81-b179-a80a82493b43
# ╠═728c16b7-50cd-43fe-a0d7-61d37952a6b3
# ╠═f7ca3ff7-b5cf-452b-b955-7219e7397324
# ╠═736920df-e4bb-4535-b982-e397aa0a782d
# ╟─c3db7eb2-356a-428f-9777-6369662d8b06
# ╠═c3a9a0ce-3450-4b17-8696-2ab8534b29f2
# ╟─4df0a18d-cb14-41b1-ba40-fd6bfcbb0b03
# ╟─2e8b1a77-1f04-4035-8d82-4061d81ecb7a
# ╟─ed5b5702-4cca-4116-a70f-4a562178f490
# ╟─771d39a5-74dc-494e-929e-1164bb08b983
# ╟─97463c54-7cc7-4497-a8a6-6422f5f582bd
# ╟─e2eca085-9f99-4e3a-9db4-e7f692aedd34
# ╟─9e898325-e9e2-45bd-af74-3dd86f00f7b5
# ╟─bcf7667f-f99b-4d10-af84-5d3879f1db5d
# ╟─2a7e5096-1e8d-4506-96d2-86de0a7d39aa
# ╟─e86a57ae-3945-4cbf-b2e4-a96f4b5295e0
# ╟─eb18303f-3dfb-4b87-90f2-f6dc542d7221
# ╟─6622f9f0-cecc-476e-9d49-7d651f433b9f
# ╟─6aa690e9-389f-4398-abae-b95060db4d90
# ╟─9cb90e76-3bb5-41ff-bc79-c4949400d904
# ╟─6712c883-b407-47e1-a666-4de05f8f8d6e
# ╟─c4bebd0d-eacf-4db4-b5b3-4dca50ab9e1b
# ╟─76e2f97b-1c06-40cd-b134-d5155aa5587d
# ╟─f8330700-e964-4e19-9c55-2b11df45789e
# ╟─91abc03b-fef7-4f93-96fc-13f1cf654f0d
# ╟─75ca478c-916f-464a-9435-8208ee726d50
# ╟─4e10271c-49f8-4f1d-869c-5fa11275d7f6
# ╟─55a09acc-84da-491c-86ba-9a66f4ea52fe
# ╟─5b4a67a9-e33e-4dc6-b9f0-fd9a2cca6f2a
# ╟─d54b5390-0ec0-4ff8-ab18-51726482ca46
# ╟─225e58ba-b78d-4a0a-be4f-ad642c879b93
# ╟─a21fbc70-9137-4d0e-8c8c-cbdc5269778f
# ╟─8eafcfed-9771-4d99-b0c5-bd75a6dab012
# ╟─6800afbf-8ac6-4308-b4cc-b37da57e42c1
# ╟─55435b26-7fc3-4c8b-8013-6fd4fb65a08e
# ╟─bcbb3db2-85b3-4cb0-9309-f5c032d14da5
# ╠═d558636d-c714-4033-ae73-5b92c3cdedf3
# ╠═70f395b2-f8c2-44d5-b0af-702659dd7fee
# ╠═b1a924f4-e2f0-445c-830f-94287a0e52f7
# ╠═95504a74-d5ef-4fb7-83a0-88914c7cbc59
# ╟─8d231f2c-4b0c-4c37-a746-16e98d4cafc8
# ╟─86101f07-67c5-4df2-911c-4013c44d6c5b
# ╠═570fa160-3adb-463e-99b8-b7dd05076908
# ╠═77f446ac-6030-48f2-9bea-93c427f9fcb9
# ╠═1faa4ab2-6c93-47dc-b631-8be52780fe7d
# ╟─e41d13ca-1dc1-45ae-9fa6-a83c4101120d
# ╟─bf563783-9784-4c74-a7b1-6d7a3ed618c5
# ╟─9ff95a9a-192b-4a12-8e2e-7acd6659c066
# ╟─5150d8f3-6e85-43f2-801a-eae5cc3e3095
# ╟─c032b3ff-c539-4e38-81d0-39b28b3a8076
# ╟─76ba4e9b-8bb0-47c4-b607-2ca711f035e6
# ╟─8c27b182-0c3c-4c19-9619-df62b7dd6bf0
# ╟─0c0c1163-0aec-4089-9acc-539b3a86d0b3
# ╟─b7583418-f4fb-4c63-b421-b5b9af269768
# ╟─d014e6aa-92f6-4ca1-be47-516565d1bb20
# ╟─6fc13413-53de-4c75-9b9e-620e0b7f8a1f
# ╟─d05e6f0f-0081-4fb6-91e9-ac2f58beda4a
# ╟─a3efd921-eb14-4901-9d6c-800cc812fe02
# ╟─b9caae1a-38aa-4d01-9cda-3d6782fb0e03
# ╟─4b61363d-87c9-4755-8286-44df34e9dd6a
# ╟─453544fc-0e3e-4e04-8c0c-192f3a038884
# ╟─c5be3956-5102-4d88-bfdb-9813c0555fe1
# ╟─92e01e21-ca77-43fc-9bf8-0c5a7aaed1bb
# ╟─18c26901-85eb-45ac-89bf-b03bd255007a
# ╟─b56e9e56-e74a-401b-b4b5-f36bb33341d5
# ╟─f2cba2aa-c541-4692-a441-e65741750a15
# ╟─5f05e717-a51a-4a99-bb11-cc493217f93f
# ╟─2a8433e3-9a3b-487b-abf3-09278ea42389
# ╟─e383bb72-49a1-4df1-84c3-b95a2ffe00f5
# ╟─f95a6de6-5e02-4237-88ba-ec44ef3d38c3
# ╟─af8194a1-a358-4cf7-b446-6b377cb76687
# ╟─79e6c4a8-cc1e-40cc-bb09-e9a7a9a8e475
# ╟─d1ba8da3-add8-4dbe-9ebf-9a32fa5cd5dd
# ╟─4dd7083a-e730-4f4b-bde8-fc1a5b08ebfc
# ╟─a5b20939-9afa-48c0-aa67-cbca6bc99804
# ╟─8d6ec2b3-997e-4df5-a3b2-c1dffa53d0ec
# ╟─25b79953-fd7c-46c1-b760-d57c09910981
# ╟─45efc71d-d5f8-474e-9b89-e72fac7110fd
# ╟─a14e505e-2e4a-4c73-8133-7560ba58916b
# ╟─9a8eef1b-27c0-4d57-a389-53708ade9058
# ╟─728f5fdf-77a5-46c7-b3ee-01064ef1b7e2
# ╟─c8923675-e73e-4621-82b9-966d8b003b97
# ╟─04e9b912-6712-4290-acc4-f24bb27a1469
# ╟─c1437dcc-22cb-424f-9b8e-326172f82d86
# ╟─8b78360a-21cb-4574-a84d-46ea4d0cedb1
# ╟─f7ca738d-5215-4e91-a2f3-a5ff10911313
# ╟─6bff7bca-ea1d-44c6-b8c3-040250f90654
# ╟─3d8add97-59e1-444a-838b-85c2a2ac60b3
# ╟─f572e113-b36b-4a6b-96c7-c26f100e1ad4
# ╠═32621224-a782-4bf6-9570-562cf2bb7360
# ╠═f6f7376e-9984-4289-b8ff-9d47e5358791
# ╠═6f72e8a5-819d-474c-a725-7f7318d964d7
# ╠═1d5b1b7c-828c-4a16-b446-cff21b015d45
# ╠═94ae440d-0644-49db-9461-f1a1ff1d7f87
# ╠═4f1d5112-dbac-4eb6-8518-0dc4193c3f8e
# ╠═85a10748-8d19-44a8-a1c5-0d13b093f1bf
# ╠═a873f760-bfc1-489f-a58e-75e12afa54f2
# ╠═d050a7ee-3aa7-4539-a236-5b6446599ded
# ╠═29474a70-32eb-4281-8626-87819afa7267
# ╠═89305cae-098f-4644-9109-d00f1e3bc04c
# ╠═c7f318b9-30e6-4b79-b7da-52f70904d246
# ╠═f1afaf8c-d9ad-446a-9826-9c4cda19993f
# ╠═01372b00-ecb2-42bd-b408-13234717d969
# ╠═f39305ea-f7f5-440e-ac55-c83e27f6e7fc
# ╠═579a203b-e6f7-4190-b874-18b00a5c3f77
# ╠═93200f46-7c8f-4362-a445-43c57b50a2d2
# ╠═7e27c349-ee76-46bd-b1c2-a9ce54974e10
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
