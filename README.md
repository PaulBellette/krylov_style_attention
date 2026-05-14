# Higher-Order Linear Attention: A Krylov-Style Recurrent Memory Experiment

This repository contains small, deliberately minimal experiments exploring a causal attention-like layer that extends linear attention with higher-order recurrent memory states.

The aim is not to propose a production-ready replacement for softmax attention. The aim is to test a narrower mechanistic question:

> If ordinary causal linear attention is a first-order recurrent memory, does adding higher-order Krylov-style memory states improve sequence propagation on tasks designed to require multi-step relational composition?

The experiments are intentionally small. They are meant to be readable, hackable, and diagnostic.

---

## Motivation

Softmax attention is extremely effective at sharp retrieval. Given queries and keys, it forms token-token scores,

$$
QK^\top
$$

then applies a row-wise softmax to produce normalized retrieval weights,

$$
Y = \mathrm{softmax}(QK^\top)V.
$$

This is powerful because it lets the model select or copy from specific previous tokens. But it also couples attention to an explicit $n \times n$ interaction matrix, which is expensive for long sequences and awkward to interpret as a compact causal state.

Linear attention takes a different route. It replaces the softmax kernel with a positive feature map,

$$
\phi(q)^\top \phi(k),
$$

and uses associativity to compute causal attention through recurrent prefix states:

$$
S_t = S_{t-1} + \phi(k_t)v_t^\top,
$$

$$
z_t = z_{t-1} + \phi(k_t),
$$

$$
y_t =
\frac{
\phi(q_t)^\top S_t
}{
\phi(q_t)^\top z_t
}.
$$

This is attractive because the sequence is summarized by a state rather than an explicit attention matrix. It is also closely related to fast-weight memory systems.

### KV cache motivation

There is also a practical systems motivation in the background: the KV cache.

In autoregressive softmax attention, generation stores keys and values for every previous token and every layer. This makes inference memory grow with context length. For long-context models, the KV cache becomes a major bottleneck.

Recent architectures such as multi-head latent attention attack this problem by compressing the per-token key/value cache. Linear attention and recurrent state-space-style models take a more radical route: instead of storing a key/value record for every previous token, they summarize the prefix into a recurrent state.

The Krylov mixer explored here belongs to that second family of ideas. In principle, it replaces an explicit per-token KV cache with a small set of recurrent moment states:

$$
S^{(0)}, S^{(1)}, \dots, S^{(m-1)}.
$$

That makes it conceptually relevant to long-context and cache-compression discussions.

However, this repository does **not** claim an efficiency win. The current implementation is a simple Python/PyTorch recurrence over sequence positions and is much slower than optimized softmax attention. The point of the prototype is only to test the mechanism:

> if we move from first-order linear-attention memory to higher-order recurrent memory, do we get useful extra propagation capacity?

Any practical KV-cache or throughput benefit would require a much more serious implementation, likely involving scan/chunking or custom kernels.


However, ordinary linear attention is only a first-order memory. It accumulates key-value moments and reads them with the current query. This raises a natural question:

> Can we build a controlled superset of causal linear attention by adding higher-order recurrent memory states?

That question led to the Krylov-style construction explored here.

---

## Core idea

Let

$$
u_t = \phi(q_t),
$$

$$
w_t = \phi(k_t),
$$

where $\phi$ is a positive feature map. In the current prototype,

$$
\phi(x) = \mathrm{softplus}(\beta x) + \epsilon,
$$

followed by L1 normalization across the feature dimension.

The zeroth-order value is just the current value vector:

$$
Y_t^{(0)} = v_t,
$$

$$
Z_t^{(0)} = 1.
$$

For each order $r \ge 1$, we maintain recurrent states

$$
S_t^{(r-1)}
$$

and

$$
g_t^{(r-1)}.
$$

The update is

$$
S_t^{(r-1)} = \lambda S_{t-1}^{(r-1)}+w_t \left(Y_t^{(r-1)}\right)^\top,
$$

$$
g_t^{(r-1)} = \lambda g_{t-1}^{(r-1)}+w_t Z_t^{(r-1)}.
$$

The next-order readout is

$$
Y_t^{(r)} = u_t^\top S_t^{(r-1)},
$$

$$
Z_t^{(r)} = u_t^\top g_t^{(r-1)}.
$$

The final output is a normalized weighted mixture of orders:

$$
y_t = \frac{\sum_{r=1}^{m} c_r Y_t^{(r)}}{\sum_{r=1}^{m} c_r Z_t^{(r)}}.
$$

The coefficient for $r=0$ is set to zero in the default experiments:

$$
c_0 = 0.
$$

This is deliberate. The transformer residual path already carries current-token information, and including $Y_t^{(0)} = v_t$ inside the normalized mixer caused the layer to collapse toward a mostly identity-like value passthrough in early tests.

The current default coefficients are geometric:

$$
c_r = \rho^{r-1}, \quad r \ge 1.
$$

---

## Relation to linear attention

This construction is best understood as a superset of causal positive-kernel linear attention.

For $m=1$, the layer reduces to ordinary causal linear attention / fast-weight memory:

$$
S_t = S_{t-1} + w_t v_t^\top,
$$

$$
g_t = g_{t-1} + w_t,
$$

$$
y_t = \frac{u_t^\top S_t}{u_t^\top g_t}.
$$

For $m > 1$, the model accumulates and reads from higher-order recurrent states. These higher-order states can be interpreted as repeated memory-read/write interactions, or informally as low-degree Krylov-style terms of an implicit causal attention operator.

So the hypothesis is not:

> Krylov attention should replace softmax attention.

The more modest hypothesis is:

> Higher-order recurrent memory states may recover some of the propagation capacity missing from first-order linear attention, while retaining a state-based causal structure.

This makes the experiments useful even if the layer is not ultimately practical. If higher-order states fail to help on favourable synthetic tasks, that says something about the practical limits of this style of linear-attention memory. If they do help, it suggests first-order linear attention may be leaving useful propagation capacity on the table.

---

## Relation to Krylov methods

Classical Krylov methods approximate matrix functions by working in the span

$$
\mathcal{K}_m(A, b) = \mathrm{span}\{b, Ab, A^2b, \dots, A^{m-1}b\}
$$

Attention can be viewed as applying a function of an implicit token-token interaction operator to values. In full softmax attention, that function is roughly exponential plus row normalization:

$$
\mathrm{softmax}(A)V,\quad A = QK^\top.
$$

The construction here is not an exact Krylov approximation to softmax. Instead, it borrows the Krylov intuition:

> rather than forming the full token-token operator, maintain recurrent states corresponding to low-order repeated interactions.

The result is a causal, positive, normalized, polynomial-like mixer.

---

## Relation to state space models

This layer also has a connection to state space models, but it is not a standard SSM.

An SSM carries a compressed recurrent state forward in time:

$$
h_t = A_t h_{t-1} + B_t x_t,
$$

$$
y_t = C_t h_t.
$$

The Krylov mixer also avoids an explicit $n \times n$ attention matrix by carrying recurrent state. The difference is that the state here is a content-addressable memory built from key/value feature moments:

$$
S_t \sim \sum_{j \le t} \phi(k_j)v_j^\top,
$$

plus higher-order extensions.

So a useful framing is:

- **Softmax attention**: sharp token-token retrieval.
- **Linear attention**: first-order content-addressed recurrent memory.
- **SSMs**: learned compressed causal dynamics.
- **Krylov mixer**: higher-order content-addressed recurrent memory.

The experiments here live mostly in the linear-attention / fast-weight-memory branch, with some conceptual overlap with SSMs.

---

## Why hybrid models?

Softmax attention and Krylov-style memory have different strengths.

Softmax attention is naturally good at sharp retrieval:

$$
\text{Which exact previous token should I copy or use?}
$$

The Krylov mixer is more naturally a diffuse recurrent state mechanism:

$$
\text{What information has propagated through the prefix state?}
$$

For that reason, one plausible architecture is hybrid:

```text
early Krylov blocks  ->  final softmax block
```

or, more generally,

```text
Krylov path:  recurrent propagation / state building
Softmax path: sharp final selection / readout
```

The current grid search includes hybrid schedules such as:

```text
krylov > softmax
krylov > krylov > softmax
```

These are not intended as final architectures. They are diagnostic probes of whether Krylov blocks are useful as state-building layers even if softmax remains better for exact readout.

---

## Synthetic tasks

The experiments use small synthetic grammars.

### Key-value retrieval

Example:

```text
<bos> C : 4 ; F : 3 ; D : 0 ; ? F = 3 <eos>
```

This tests sharp associative retrieval. It is expected to favour softmax attention.

The model must retrieve the value paired with the queried key.

### Path / relation composition

Example:

```text
<bos> A > B ; B > C ; C > D ; ? A = D <eos>
```

For a fixed hop count, the model must follow a chain of relations.

This task is intended to test whether higher Krylov order helps with multi-step propagation.

For example:

- hop 1 should require direct relation lookup.
- hop 2 should require one composition.
- hop 3 should require two compositions.

A favourable result for the Krylov hypothesis would be:

```text
hop 2: m=2 improves over m=1
hop 3: m=3 improves over m=1 or m=2
```

Especially if the improvement persists across random seeds and depth settings.

---

## Experimental questions

The current experiments are organized around a few simple questions.

### 1. Can the layer train at all?

Early tests showed that pure Krylov attention can learn simple hop-1 relation lookup, though slower than softmax.

### 2. Does higher order help?

The main comparison is:

```text
m = 1  vs  m = 2  vs  m = 3
```

where $m=1$ is the linear-attention-like case.

### 3. Does Krylov order reduce required depth?

A single layer may be too constrained for multi-hop reasoning, even for softmax. A more realistic question is whether higher-order Krylov states reduce the amount of transformer depth or the number of training steps needed.

### 4. Are hybrids better than pure Krylov?

The working hypothesis is that pure Krylov may struggle with sharp final selection, while a final softmax layer may recover exact readout from a state built by earlier Krylov layers.

---

## Setup

This is a single-file uv script.

```bash
uv run tinygpt_krylov_grid.py --grid quick --device cpu
```

## Running experiments

A typical grid run:

```bash
uv run tinygpt_krylov_grid.py \
  --grid path_depth \
  --device cuda \
  --steps 30000 \
  --eval-every 250 \
  --log-every 250 \
  --early-stop \
  --target-acc 0.99 \
  --grammar path \
  --seq-len 20 \
  --n-edges 3 \
  --hide-hop-token \
  --d-model 128 \
  --n-heads 4 \
  --d-ff 512 \
  --batch-size 64 \
  --eval-batches 20 \
  --grid-depths 2,3,4 \
  --grid-hops 2,3 \
  --grid-m-values 1,2,3 \
  --grid-archs softmax,krylov,krylov_then_softmax \
  --grid-seeds 1,2,3 \
  --krylov-rho 1.0 \
  --krylov-beta 0.5 \
  --krylov-state-decay 1.0 \
  --output-dir runs/path_depth_grid
```

The script writes:

```text
results.csv
results.jsonl
manifest.json
```

The most important fields are:

- `hop`
- `n_layers`
- `schedule`
- `krylov_m`
- `seed`
- `best_acc`
- `solved`
- `solved_step`

---

## Results so far

The current results are deliberately treated as diagnostic rather than conclusive. The main value of the experiments is not that they produce a clean win, but that they expose where the proposed mechanism helps, where it does not, and where the synthetic task itself has high variance.

### Key-value retrieval

The key-value retrieval task favours sharp associative lookup. This is the natural strength of softmax attention.

In early tests, softmax learned the simplified key-value task reliably, while pure Krylov layers were much less reliable. This is consistent with the expected weakness of a diffuse recurrent memory when the task is essentially:

```text
find the exact previous key and copy its paired value
```

This result is useful because it rules out an overly broad claim. The Krylov-style mixer should not be described as a drop-in replacement for softmax attention on sharp retrieval tasks.

### Hop-2 path composition

The clearest positive signal appears on the hop-2 path task.

For pure Krylov stacks, increasing order from $m=1$ to $m=2$ or $m=3$ reduced the number of training steps needed to solve the task. The effect was strongest at shallow depth.

Representative median solved steps:

```text
hop=2, depth=2, pure Krylov
m=1: 10000
m=2:  6250
m=3:  5750

hop=2, depth=3, pure Krylov
m=1: 8000
m=2: 7500
m=3: 7250

hop=2, depth=4, pure Krylov
m=1: 7750
m=2: 6750
m=3: 6500
```

This is consistent with the motivating hypothesis:

> higher-order recurrent memory states can provide useful propagation capacity beyond the $m=1$ linear-attention-like case.

However, softmax remained faster on the same task:

```text
hop=2 softmax median solved steps
depth=2: 2250
depth=3: 2750
depth=4: 1750
```

So the result is not that Krylov-style attention beats softmax. The more accurate conclusion is:

> on this small hop-2 composition task, higher Krylov order improves over first-order linear-style memory, but softmax remains the stronger and more reliable baseline.

### Hop-3 path composition

A first hop-3 run was not diagnostic because the task allowed a shortcut: with `n_edges=3` and `hop=3`, the answer was always the final node of the chain. Models solved this almost immediately, so those results should be ignored.

A corrected hop-3 run used more edges than the hop count, so that the start position varied. This removed the shortcut and made the task much harder.

The corrected hop-3 results were ambiguous:

```text
hop=3, n_edges=5, pure Krylov

depth=2:
  m=1: 0/3 seeds solved
  m=2: 1/3 seeds solved
  m=3: 1/3 seeds solved

depth=3:
  m=1: 1/3 seeds solved
  m=2: 1/3 seeds solved
  m=3: 1/3 seeds solved

depth=4:
  m=1: 1/3 seeds solved
  m=2: 0/3 seeds solved
  m=3: 0/3 seeds solved
```

Softmax was still the strongest baseline at higher depth:

```text
hop=3, n_edges=5, softmax

depth=2: 0/3 seeds solved
depth=3: 1/3 seeds solved
depth=4: 2/3 seeds solved
```

The hop-3 results do not support a simple conclusion either way. Higher-order Krylov solved some shallow-depth seeds that $m=1$ did not, which is a weak positive signal. But the effect was not robust across depth or random seed, and at depth 4 the higher-order variants were worse than $m=1$ in this small run.

Accuracy also tended to plateau near rough fractions such as one-third and two-thirds. This suggests that models may be learning some start positions or positional heuristics rather than a general path-composition algorithm.

The honest interpretation is:

> hop-3 exposes high variance and brittle optimization. The experiments show that higher-order Krylov states change the learning dynamics, but they do not yet show a robust advantage.

### Hybrid schedules

Hybrid schedules such as

```text
krylov -> softmax
krylov -> krylov -> softmax
```

were included as probes of a possible division of labour:

```text
Krylov blocks:  diffuse propagation / state building
Softmax block:  sharp final readout
```

The results were mixed. Some hybrid runs were competitive, but the pattern was not consistent enough to claim that hybrids are better than either pure softmax or pure Krylov.

The hybrid idea remains plausible, but the current evidence is not strong.

### Current interpretation

The main result is ambiguous in the right way.

The experiments support a modest claim:

> the proposed layer is a real superset of causal positive-kernel linear attention, and higher-order states can help on some favourable composition tasks.

They do not support a stronger claim:

> higher-order Krylov-style attention is a generally better replacement for softmax attention.

A fair summary is:

```text
m=1 gives a linear-attention-like recurrent memory.
m>1 adds a real extra mechanism.
That mechanism can help on hop-2 composition.
On harder hop-3 composition, variance dominates and conclusions are unclear.
Softmax remains much more reliable for sharp retrieval and final selection.
```

This is still a useful outcome. It suggests the construction is interesting as a mechanistic probe and possibly as an inductive bias, but not yet as a practical attention replacement.

---

## Practical limitations

This implementation is intentionally simple and not optimized.

The Krylov layer is currently implemented as a causal recurrent loop over sequence positions. This is convenient for testing, but it is much slower than optimized softmax attention.

This matters especially for the KV-cache motivation. In principle, a recurrent state-based layer avoids storing a per-token key/value cache. In this prototype, that theoretical advantage is irrelevant in practice because the recurrent loop is the bottleneck. To make the cache argument meaningful, the recurrence would need to be implemented efficiently.

A practical implementation would likely need one or more of:

- batched tensorized recurrence;
- parallel scan / chunked scan formulation;
- custom CUDA/Triton kernels;
- a more careful hybrid architecture;
- learnable coefficients $c_r$;
- gating or normalization improvements.

The current code is a mechanism test, not an efficiency claim.

---

## Further work

### Learned coefficients

The current experiments use fixed geometric coefficients,

$$
c_r = \rho^{r-1}.
$$

This is simple and stable, but probably too rigid. A natural next step is to learn the order coefficients directly, perhaps with a positivity constraint such as softplus or a simplex normalization.

Useful variants include:

- one learned coefficient vector per layer;
- one learned coefficient vector per head;
- input-dependent gates over orders;
- initialization near the current geometric schedule.

This would test whether the model actually wants higher-order terms, and whether it wants them uniformly across layers.

### Efficient recurrence

The current implementation loops over sequence positions in Python/PyTorch. That is good for readability and bad for speed.

Any real KV-cache benefit would require making the recurrent update efficient. Possible directions include:

- batching the state updates across batch and heads;
- expressing the recurrence as a scan;
- chunking the sequence and composing chunk states;
- writing a custom Triton or CUDA kernel;
- exploring whether a restricted update rule admits a parallel prefix formulation.

Until this is done, the implementation should not be treated as a practical long-context architecture.

### Different task regimes

The current synthetic tasks emphasize retrieval and symbolic path composition. These are useful stress tests, but they may not be the best use case for the layer.

The Krylov-style mixer may be more useful as a global mixing layer or state-building layer than as a pure readout mechanism. In particular, it may be better suited to tasks where diffuse global context is useful and exact token retrieval is less central.

Possible follow-up settings:

- algorithmic tasks with soft global state rather than exact copying;
- long-range smoothing or aggregation tasks;
- sequence classification with long contextual dependencies;
- hybrid transformer blocks where Krylov mixing complements rather than replaces softmax attention;
- architectures that use Krylov-style layers early and softmax layers late.

The more realistic question may not be:

> can this replace softmax?

but rather:

> does this provide a useful inductive bias for global recurrent mixing?

### Better diagnostics

The hop-3 results suggest that models may solve only some start positions. Future evaluations should report accuracy broken down by path start index, hop count, and answer position.

This would distinguish genuine composition from positional heuristics.

---

## Caution on claims

This project does **not** claim that Krylov attention beats softmax attention.

A more accurate claim is:

> This is a small experimental family that contains causal positive-kernel linear attention as the $m=1$ case and adds higher-order recurrent memory states. The experiments test whether those higher-order states help on synthetic tasks designed to reward multi-step propagation.

That is the intended scope.

---

## Acknowledgements

This project grew out of a collaborative exploration between Paul Bellette and ChatGPT.

The initial motivation came from discussing multi-head latent attention, low-rank factorization, linear attention, Taylor/Krylov approximations to attention-like operators, and the relationship between attention and state-space models.

Paul drove the research direction, experimental judgement, implementation testing, and interpretation. ChatGPT contributed derivations, prototype code, debugging assistance, experiment design suggestions, and README drafting.

The project is part of a broader pattern of small, curiosity-driven experiments in machine intelligence and numerical structure. The goal is not just to build a layer, but to understand what kinds of sequence mechanisms are actually useful when reduced to small, testable forms.
