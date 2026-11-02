# Curriculum

Target: about 370 sheets. Each part is written by one writer, in parallel with the others, so the boundaries below matter: teach what your part owns, and when you need something another part owns, refer to it ("Part 2 derives this") in one sentence instead of re-teaching it.

Sheet counts are targets (±10%). Every bullet is at least one sheet unless it says otherwise. Order within a chapter is the teaching order. Each part starts with a part-opener sheet (`layout: "part"`, counted in the total) and each chapter ends with a consolidating sheet or a `check`.

The reader knows systems deeply and machine learning not at all (see AUTHORING.md §1).

---

## Part 0 · Orientation (6, written)
Cover, the running example, the model, how to read a sheet, the route, notation.

---

## Part 1 · The math toolkit (≈ 44) · prefix `math-` · file `content/01-math.js`
Goal: every piece of math the later parts use, built from school algebra, each with numbers you can check by hand. Interactive plots everywhere.

**1.1 Functions (5).** A function as a rule from inputs to outputs, shown as a table of values and a graph · linear functions y = wx + b: slope and intercept, with sliders · composing functions f(g(x)) with a worked number · functions of two inputs as a surface and as a contour map (level curves) · the functions that recur in this deck: x², eˣ, ln x, max(0, x), 1/(1+e⁻ˣ), plotted.

**1.2 Derivatives (8).** Average rate of change between two points (secant slope) · shrinking the gap: the slope of x² at x = 3 for h = 1, 0.1, 0.01, 0.001 (7, 6.1, 6.01, 6.001) → derivative as a limit · the tangent line, with a draggable point · rules: power, constant factor, sum, with the derivative of 3x² + 2x · derivatives of eˣ and ln x · numerical derivative (finite differences), central vs forward difference error, and float32 round-off when h is too small (show the U-shaped error curve) · the sign of the derivative says which way to move to decrease a function; the minimum is where it is 0 · second derivative as curvature (one sheet, used later for learning-rate limits).

**1.3 The chain rule (5).** Derivative of a composition, worked on (3x + 1)² both by expanding and by the chain rule · a chain of three functions as a product of local slopes, with numbers at each stage · drawing the computation as a graph of operations (the `graph` figure: forward values, then local derivatives multiplied backwards) · a variable used twice: sum over paths (d/dx of x·x and of sin(x)·x) · why this matters: a neural network is a long chain, and Part 3 runs this rule backwards through it.

**1.4 Partial derivatives and the gradient (5).** Partial derivative: vary one input, hold the others · the gradient vector of f(x, y) = x² + 3y², worked at a point · the gradient points in the direction of steepest increase, perpendicular to level curves (contour plot with arrows) · its length is the slope in that direction; directional derivative = gradient · unit direction · gradient of a function of many inputs (n = 9,968 for our model).

**1.5 Vectors (5).** A vector as a list of numbers; our model represents each token as 16 numbers (show a real embedding row) · addition and scaling, drawn · dot product as sum of products, worked; geometric meaning |a||b|cos θ with the `vectors` figure · length (norm) and cosine similarity; similarity between two real token embeddings of our model · many vectors stacked as rows: a matrix.

**1.6 Matrices (7).** A matrix as a grid; shapes m × n · matrix–vector product as one dot product per row, worked · as a transformation of the plane (rotation, scaling, shear) with a draggable 2 × 2 matrix · matrix–matrix product with the `matmul` figure; the shape rule (m × k)(k × n) = m × n · cost: m·n·k multiply-adds = 2mnk FLOPs, counted for our model's 8 × 16 by 16 × 113 output layer · transpose and the identity · a batch of inputs as rows of one matrix: one matmul processes every token at once (why GPUs matter, forward pointer to Part 6).

**1.7 Probability (6).** A probability distribution over a finite set; it sums to 1; the 3 tables in our grammar · conditional probability p(next | prefix) with the grammar's exact numbers (after SELECT, p(name) = 1/12) · the chain rule of probability: p(whole query) = product of next-token probabilities, computed for our query (1/324 from the grammar, and from the model) · expectation (average weighted by probability) · sampling: turning a uniform random number into a draw with the cumulative distribution (interactive) · independence and the law of large numbers (sample mean converges; one sheet).

**1.8 Exponentials and logarithms (4).** eˣ and its growth; ln as its inverse · logs turn products into sums: log p(query) as a sum of 9 terms · the product of 1,000 probabilities of 0.1 underflows float32 (smallest normal ≈ 1.2 × 10⁻³⁸) while the sum of logs is −2,302.6 · log-sum-exp and why subtracting the max keeps exponentials finite (worked with logits 1000, 1001, 1002).

Owns no widgets.

---

## Part 2 · Learning from data (≈ 36) · prefix `learn-` · file `content/02-learning.js`
Goal: what it means to fit a model to data: loss functions, gradient descent, classification with softmax and cross-entropy, information measures, generalization. All on small examples the reader can do by hand, then on the real model.

**2.1 Models and parameters (4).** A model is a function with adjustable numbers (parameters) · a tiny dataset (5 points) and the line ŷ = wx + b · parameters vs inputs vs hyperparameters · learning = choosing parameters that make predictions match data; the loop in one picture (predict, measure, adjust).

**2.2 Loss functions (7).** The error of one prediction · squared error, and why squared: no sign cancellation, smooth, big errors cost more (residual squares on the `plot` figure) · mean squared error over the dataset, worked · absolute error and outliers: one bad point moves the MSE line much more than the MAE line (interactive) · loss as a function of the parameter: the curve L(w) for fixed b · two parameters: the loss surface L(w, b) as a contour map, with the minimum marked · the exact minimum for linear regression (set derivatives to zero, solve; worked), and why for neural networks there is no formula, only search.

**2.3 Gradient descent (8).** The update w ← w − η · dL/dw, derived from "move against the slope" · three steps by hand on L(w) = (w − 3)² with η = 0.1 (−1 → −0.2 → 0.44 → 0.952) · learning rate too small, right, too large, diverging (the `gd` figure; for this L, η > 1 diverges, explain why from the update factor 1 − 2η) · gradient descent on two parameters (fit w and b together) · narrow valleys: on L = w₁² + 10w₂² the largest stable η is set by the steepest direction (η < 0.1), so progress along the flat one is slow (zig-zag) · stochastic gradient descent: the gradient on a random minibatch is an unbiased, noisy estimate; batch, step, epoch defined with numbers · noise vs batch size (the standard deviation of the gradient estimate shrinks as 1/√B, shown with sampling) · when to stop; what a training curve looks like.

**2.4 Classification (8).** Predicting a category instead of a number · scores (logits) to probabilities: the sigmoid for two classes, worked · softmax for many classes, worked on four logits; invariant to adding a constant · our model's real logits after `… WHERE` and their softmax (owns the `logits`? no: compute directly with `TinyLM.forward`; show the 4 column tokens) · why squared error is a poor loss for probabilities (flat gradient when confidently wrong) · likelihood of the data; negative log-likelihood; cross-entropy loss = −ln p(correct class), with values: p = 0.9 → 0.105, p = 0.5 → 0.693, p = 0.1 → 2.303, p = 0.01 → 4.605 · the gradient of softmax + cross-entropy with respect to the logits is p − y, derived step by step (`steps` figure) · gradient descent on four logits with the real data distribution (widget `gradient`).

**2.5 Information (4).** Entropy: average surprise −Σ p ln p; a fair coin (0.693 nats = 1 bit), a fair 9-sided die (ln 9 = 2.197), a certain outcome (0) · cross-entropy and KL divergence: cross-entropy = entropy + KL ≥ entropy, so the best possible average loss is the entropy of the data · our grammar's entropy per position, and the floor 0.6937 nats per token averaged over the 8 predictions (forward pointer: Part 4 shows the model reaching 0.6942) · perplexity = e^loss; a perplexity of 9 means "choosing among 9 equally likely options".

**2.6 Generalization (5).** Training data vs held-out data · overfitting with polynomials: degree 1, 3, 9 on 10 noisy points; training loss goes to 0, held-out loss goes up (interactive) · underfitting · validation sets and early stopping · regularization: L2 weight decay adds λ‖w‖² and shrinks weights, with numbers; why large models trained on huge data barely overfit in one pass (forward pointer to Part 5's held-out experiment).

Owns widget: `gradient`.

---

## Part 3 · Neural networks (≈ 42) · prefix `nn-` · file `content/03-networks.js`
Goal: from one neuron to deep networks, backpropagation in full, PyTorch autograd, and the techniques that make deep networks trainable (initialization, normalization, residuals, Adam, schedules).

**3.1 Neurons (5).** Linear models cannot draw curved boundaries: XOR, shown · a neuron: weighted sum + bias, then a nonlinearity · activation functions plotted with their derivatives: step, sigmoid, tanh, ReLU, GELU (and why the step function's zero derivative breaks learning) · a stack of linear layers is still linear: W₂(W₁x) = (W₂W₁)x, worked with 2 × 2 matrices · two ReLU neurons solve XOR, with the weights.

**3.2 Layers and networks (6).** A layer = matrix multiply + bias + activation, applied to a batch at once · a multi-layer perceptron (MLP), its shapes, written as code · a forward pass with numbers through a 2-3-1 network (`network` figure) · sums of ReLUs build any curve: fit a 1-D function with k bumps (interactive, k = 1…8) · parameter counts: a d_in × d_out layer has d_in·d_out + d_out; our model's MLP 16 → 64 → 16 (2,048 weights, no biases) · depth vs width.

**3.3 Backpropagation (10).** The goal: dL/dw for every weight, at a cost close to one forward pass · the forward pass records values (computational graph) · the backward pass multiplies local derivatives from the loss back (`graph` figure on (w·x + b − y)²) · one neuron by hand, every number · a layer: dL/dW = xᵀ δ (an outer product), dL/dx = δ Wᵀ, dL/db = δ, derived with shapes · through ReLU: multiply by the 0/1 mask · through softmax + cross-entropy: p − y (refer to Part 2) · a full 2-layer example with numbers, checked against finite differences · cost: the backward pass is about 2× the forward pass (two matmuls per weight matrix), so training costs ≈ 3× inference per token, the origin of 6 in 6ND · memory: every activation is kept for the backward pass; count it for a 2-layer MLP and a batch.

**3.4 Autograd and PyTorch (5).** Tensors, shapes and dtypes · `requires_grad`, the recorded graph, `loss.backward()` · the training loop in PyTorch (code figure with notes on every line: zero_grad, forward, loss, backward, step) · what `optimizer.step()` does to each parameter · gradient accumulation across micro-batches (why `zero_grad` exists).

**3.5 Making deep networks train (11).** Initialization: all-zero weights keep neurons identical (symmetry); variance through 10 layers explodes or vanishes unless scaled by 1/√fan_in (interactive, show activation std per layer) · Xavier and He initialization · vanishing and exploding gradients through depth · normalization: BatchNorm vs LayerNorm vs RMSNorm, with a worked RMSNorm on a real vector of our model · residual connections: y = x + f(x) gives the gradient a direct path; gradient norms with and without · momentum (worked on the narrow valley) · RMSProp and per-parameter step sizes · Adam: the two moving averages, bias correction, worked for 3 steps · AdamW: decoupled weight decay · learning-rate warmup and cosine decay (our model's schedule: 100 warmup steps, 0.01 → 0.001) · gradient clipping by global norm (widget `optim` on one of these sheets).

**3.6 A network trained in the page (5).** Train a small MLP in the browser on a 2-D classification problem (two spirals or moons) with a custom figure: live loss curve and decision boundary, controls for width, depth, learning rate · what changes when you change each · overfitting a small network · the same loop, scaled, trains every LLM (bridge to Part 4).

Owns widget: `optim`.

---

## Part 4 · Language modeling (≈ 34) · prefix `lm-` · file `content/04-language.js`
Goal: turn text into numbers, define next-token prediction precisely, count-based models, neural language models with embeddings, and decoding. Uses the running example on nearly every sheet.

**4.1 Text as numbers (5).** Characters, Unicode code points, UTF-8 bytes (bytes of "é", "日", an emoji) · vocabularies and ids · one-hot vectors and why they are wasteful (113-dimensional for us, 128,256 for Llama 3) · why not whole words (vocabulary size, unseen words) · subwords as the compromise.

**4.2 Tokenization (7).** Byte-pair encoding: count adjacent pairs, merge the most frequent, repeat; worked on a few strings · BPE on our corpus, merge by merge (widget `bpe`) · encoding new text with the learned merges · decoding back to text · byte-level BPE (start from 256 bytes so any text encodes) · vocabulary size trade-off: a bigger vocabulary means shorter sequences but a bigger embedding table and softmax; numbers for 32K vs 128K vocabularies · tokenization quirks: leading spaces, digits, other languages cost more tokens; special tokens (BOS, EOS, padding).

**4.3 Next-token prediction (6).** The task: given a prefix, a distribution over the next token (widget `predict`) · the chain rule of probability makes this a complete model of text · one sequence gives T training pairs by shifting by one; our query gives 8 (widget `learn`) · the loss is the average cross-entropy over positions (refer to Part 2) · the loss per position for our model and the floor (widget `loss`) · perplexity of our model per position.

**4.4 Counting models (5).** The bigram model: count pairs in the corpus, normalize rows (build our bigram table from the 2,000 tokenizer-corpus queries, which you can recompute from `TinyLM.sampleQuery` with the same distribution or from the grammar exactly) · sampling from a bigram model; its loss on our grammar (it cannot know which table matches the column) · n-grams with longer context; the number of possible contexts grows as Vⁿ (113⁴ ≈ 1.6 × 10⁸ for n = 5) · smoothing for unseen contexts · what a model needs that counting cannot give: sharing statistics between similar contexts.

**4.5 Neural language models (6).** Embeddings: a learned table of vectors; looking up row id = one-hot × matrix (widget `embed`) · what the trained embeddings of our model encode (similar roles end up near each other; use real cosine similarities) · a fixed-window neural LM (Bengio et al. 2003): concatenate k embeddings, MLP, softmax · the output layer: logits = h·W_U over the whole vocabulary; its cost for a 128K vocabulary · parameter counts of embedding and output tables (113 × 16 = 1,808 each for us; 128,256 × 4,096 = 525M for Llama 3 8B) · the limits of a fixed window, motivating attention (forward pointer to Part 5).

**4.6 Generating text (5).** The generation loop: predict, choose, append, repeat (widget `generate`) · greedy decoding and its failure modes · temperature, worked on real logits · top-k and top-p (nucleus) sampling, worked · cost: one forward pass per generated token; why that makes serving different from training (forward pointer to Part 8).

Owns widgets: `world` (optional), `bpe`, `predict`, `learn`, `loss`, `embed`, `generate`.

---

## Part 5 · The transformer (≈ 45) · prefix `tf-` · file `content/05-transformer.js`
Goal: the full architecture computed on our query with the trained weights, every intermediate number available, then the modern variants used by Llama-class models.

**5.1 Why attention (4).** Each prediction needs information from specific earlier positions at varying distances (widget `need`) · recurrent networks process tokens one at a time (sequential, lossy state); one sheet, factual · attention: every position computes a weighted average of earlier positions' vectors, with weights computed from the data · the plan of a transformer in one diagram.

**5.2 Self-attention step by step (10).** Input = token embedding + position embedding, for our 8 positions (real numbers) · queries, keys, values: three linear maps of the same vector (shapes 16 → 8 per head) · scores q·k for position 4 (`WHERE`) against positions 0–4 (widget `qkv`) · scaling by √d_head: the variance of a dot product of random vectors grows with d (demonstrate) · the causal mask: −∞ above the diagonal so a position cannot see the future · softmax turns scores into weights; weighted sum of values (widget `mix`) · the whole computation in matrix form softmax(QKᵀ/√d + M)V with every shape for our model · the full 8 × 8 attention matrix of layer 0 head 0 (heat figure, real numbers) · cost: T²·d per head for scores and for mixing; at T = 8 it is tiny, at T = 128K it dominates (numbers) · a check.

**5.3 Multi-head attention (5).** Split d into heads that attend independently · attention patterns of our two heads per layer (widget `heads`) · concatenate heads, output projection W_O · parameter count 4d² per attention block (1,024 for us) · what our trained heads attend to (describe from real patterns, e.g. the head that copies the table name).

**5.4 The transformer block (8).** The residual stream: each block adds to x rather than replacing it (widget `block`) · pre-norm RMSNorm, worked on a real vector · the MLP sublayer d → 4d → d with GELU · one block as a diagram with every shape · stacking blocks; what changes between layer 0 and layer 1 · the final norm and unembedding to 113 logits (widget `logits`) · parameter count: 12·L·d² + V·d (+ positions, norms), checked against 9,968 term by term · FLOPs per token ≈ 2 × parameters (+ attention), and for our query.

**5.5 Positions (4).** Without positions, attention is permutation-invariant (show: shuffle the inputs, same set of outputs) · learned absolute positions (ours, 8 × 16) · sinusoidal positions · rotary embeddings (RoPE): rotate each 2-D pair of q and k by an angle proportional to the position, so q·k depends on the distance; worked 2-D example.

**5.6 Training the transformer (7).** Our training run: loss curve and what the model predicts at each snapshot (widget `train`) · which facts it learns first (grammar keywords, then table/column consistency) · the backward pass through the last layers (widget `backprop`) · the final loss 0.6942 vs the floor 0.6937 · memorization vs generalization: the held-out model (widget `heldout`) · how big real models are configured (widget `real`) · a check.

**5.7 Modern variants (7).** SwiGLU feed-forward and its parameter count · grouped-query and multi-query attention: fewer key/value heads (Llama 3 70B: 64 query heads, 8 KV heads) and what it saves (forward pointer to KV cache in Part 8) · mixture of experts: a router picks the top-k experts per token (widget `moe`); total vs active parameters (e.g. Mixtral 8×7B: 46.7B total, 12.9B active; DeepSeek-V3: 671B total, 37B active) · long context: attention cost and sliding-window attention · weight tying (sharing embedding and unembedding) · the Llama 3 architecture table, every hyperparameter explained · a consolidating sheet: our model vs Llama 3 8B, line by line.

Owns widgets: `need`, `qkv`, `mix`, `heads`, `block`, `logits`, `train`, `backprop`, `heldout`, `real`, `moe`.

---

## Part 6 · GPUs and CUDA (≈ 48) · prefix `gpu-` · file `content/06-gpu.js`
Goal: the reader can explain how an H100 executes a matrix multiply, write and reason about CUDA kernels, predict whether an operation is compute- or memory-bound, and explain what FlashAttention and kernel fusion change. This part was missing from the first version and the reader asked for it explicitly. Go deep; this reader has an HPC/FPGA background.

**6.1 Why GPUs (5).** Where an LLM's time goes: matrix multiplies dominate FLOPs (count for one Llama 3 8B layer) · a CPU core vs a GPU SM: few wide out-of-order cores with large caches vs many simple lanes that hide latency with parallelism · throughput vs latency design, with numbers (a server CPU's peak FP32 vs an H100's) · the H100 SXM headline numbers: 132 SMs, 989 TFLOP/s BF16 dense, 3.35 TB/s HBM3, 80 GB, 50 MB L2, NVLink 900 GB/s (verify each against NVIDIA's datasheet / whitepaper) · arithmetic intensity: FLOPs per byte moved, defined with a vector add (1 FLOP per 12 bytes).

**6.2 Inside an H100 (8).** Chip hierarchy: GPCs → TPCs → SMs, memory controllers, L2 (diagram) · inside an SM: 4 processing blocks, each with a warp scheduler, dispatch, a 64 KB register file slice (16,384 32-bit registers), 32 FP32 lanes, 16 INT32, 16 FP64 (check), 1 tensor core; 256 KB combined L1/shared memory per SM (up to 228 KB shared) · CUDA cores vs tensor cores: FP32 SIMT throughput vs matrix throughput (67 vs 989 TFLOP/s, check) · the memory system: registers, shared memory/L1, L2, HBM, with capacity, bandwidth and latency in cycles (cite measurements, e.g. microbenchmark papers, and say they are approximate) · HBM: stacks, bus width, why bandwidth grew slower than FLOPs (H100 vs A100 vs V100 ratio table) · clock, power, and why "peak" is rarely reached · NVLink and NVSwitch inside a server (8 GPUs), PCIe to the host · a datasheet sheet comparing V100, A100, H100, B200 (verify each number; mark dense vs sparse).

**6.3 The CUDA programming model (8).** Host and device, separate memories · a kernel is a function run by many threads; `__global__` and the `<<<grid, block>>>` launch · threads, blocks, grids, and the index formula `blockIdx.x * blockDim.x + threadIdx.x` (custom interactive grid figure: pick grid and block sizes, see which thread handles which element) · vector add, line by line (`code` figure), including the bounds check · allocating and copying memory (`cudaMalloc`, `cudaMemcpy`) and what a PCIe copy costs compared with the kernel · 2-D grids for matrices; row-major indexing · blocks run independently in any order; `__syncthreads()` synchronizes within a block only · streams and asynchronous execution (one sheet).

**6.4 How an SM executes threads (8).** Warps: 32 threads that issue together (SIMT) · the warp scheduler switches between resident warps every cycle to hide memory latency (timeline figure) · how much parallelism is needed: Little's law, bytes in flight = bandwidth × latency, worked for HBM · branch divergence: both sides of an if/else run with lanes masked (interactive: choose the branch pattern, see the cost) · occupancy: registers, shared memory and block size limit resident warps; calculator (registers per thread → max warps per SM, with 65,536 registers and 64 warps max) · when higher occupancy does not help (ILP, latency-bound vs bandwidth-bound) · warp-level primitives (shuffle, reductions) for a sum · a parallel reduction kernel, step by step.

**6.5 Memory access (8).** Global memory transactions and coalescing: 32 threads reading 32 consecutive floats = 128 bytes in 4 sectors; strided access multiplies the traffic (custom interactive figure: choose a stride, count sectors) · row-major vs column-major access in a matrix transpose · shared memory as a programmer-managed cache · bank conflicts: 32 banks of 4 bytes; a column access in a 32 × 32 tile conflicts 32-way; padding to 33 fixes it (custom interactive figure) · the roofline model: attainable FLOP/s = min(peak, intensity × bandwidth); H100 ridge point ≈ 295 FLOP/byte for BF16 (widget `gpu` on this sheet) · placing real operations on the roofline: elementwise add, softmax, LayerNorm, a 4096³ matmul, decode-time matrix–vector product · memory-bound kernels are optimized by moving fewer bytes, compute-bound ones by using tensor cores · a check.

**6.6 Matrix multiplication on a GPU (7).** The naive kernel: one thread per output element, 2K loads per output, intensity ≈ 0.25 FLOP/byte (count it) · tiling with shared memory: load a tile of A and of B once, reuse it T times; intensity grows with the tile size (custom animated figure stepping through tiles of a small matmul) · register tiling: each thread computes a small block of outputs · tensor cores: warp-level matrix-multiply-accumulate on small fragments (e.g. m16n8k16 in BF16 with FP32 accumulate; verify shapes in PTX docs), and the Hopper additions (wgmma, TMA) in one sheet · how close cuBLAS gets to peak, and why shapes matter: tile quantization and wave quantization with numbers (e.g. 132 SMs) · CUTLASS and the structure of a production GEMM (one sheet) · matmuls in our model are far too small to fill an H100: count the FLOPs and the launch overhead.

**6.7 Kernels for transformers (7).** Kernel fusion: an unfused chain of elementwise ops reads and writes HBM per op; count bytes for bias + GELU + dropout, fused vs unfused · a numerically stable softmax kernel; online softmax (running max and sum in one pass), worked on numbers · FlashAttention: tile Q, K, V into SRAM, never write the T × T matrix, rescale partial results with the online softmax; HBM traffic O(T²) vs O(T²d²/M) (widget `flash`) · FlashAttention-2/3 changes in one sheet (work partitioning, Hopper features) · Triton: writing a tiled kernel in Python (code figure: a fused softmax or matmul) · `torch.compile` and CUDA graphs: removing launch overhead for small kernels · profiling: reading an Nsight Systems timeline (kernel gaps, memcpy, NCCL), and Nsight Compute's achieved bandwidth and occupancy (timeline figure) · a consolidating sheet: where every microsecond of one transformer layer goes on an H100.

Owns widgets: `gpu`, `flash`.

---

## Part 7 · Training at scale (≈ 42) · prefix `scale-` · file `content/07-scale.js`
Goal: what it takes to train a model like Llama 3.1 405B: the compute budget, the memory of a training step, numerical formats, collective communication (the reader's PhD area: be exact), every form of parallelism, and operating a run.

**7.1 The compute budget (5).** Training FLOPs ≈ 6·N·D, derived from 2 (forward) + 4 (backward) per parameter per token (widget `flops`) · our model vs Llama 3.1 405B (3.8 × 10²⁵ FLOPs) · model FLOPs utilization (MFU): definition and worked numbers; time and GPU count for a run (calculator) · scaling laws: loss as a power law in N and D; Chinchilla's ≈ 20 tokens per parameter (widget `scaling`) · training past compute-optimal to make inference cheaper (Llama 3 8B on 15T tokens).

**7.2 Memory of a training step (6).** What has to be stored: weights, gradients, optimizer state (Adam's two moments), master weights: 16 bytes per parameter with mixed precision (widget `memory`) · activation memory per layer per token (count the tensors in one block of our model, then the published formula for large models) · activation checkpointing: recompute in the backward pass, trade about 33% more compute for memory · a 70B model needs ≈ 1.1 TB for states alone: it must be split across GPUs · micro-batching and gradient accumulation to reach a large global batch.

**7.3 Number formats (5).** Floating point: sign, exponent, mantissa; FP32, TF32, FP16, BF16, FP8 (E4M3, E5M2) bit layouts, range and precision (widget `precision`) · rounding error and what it does to a sum of many small numbers (accumulate in FP32) · FP16 needs loss scaling; BF16 does not (same exponent range as FP32) · the mixed-precision recipe: BF16 matmuls, FP32 master weights and accumulation · FP8 training: per-tensor and fine-grained (block) scaling (DeepSeek-V3).

**7.4 Collective communication (7).** Point-to-point vs collective operations · broadcast, reduce, all-reduce, reduce-scatter, all-gather, all-to-all, defined with data diagrams · ring all-reduce = reduce-scatter + all-gather; each GPU sends 2(N−1)/N × the data; step-by-step figure · tree and hierarchical algorithms; latency vs bandwidth terms (α–β cost model) · NCCL's algorithm bandwidth vs bus bandwidth · interconnects: NVLink/NVSwitch inside a node (900 GB/s per H100), InfiniBand/RoCE between nodes (400 Gb/s per NIC), rail-optimized topologies · overlapping communication with computation (timeline).

**7.5 Data parallelism (5).** Replicate the model, split the batch, all-reduce the gradients (widget `allreduce`) · gradient bucketing and overlap with the backward pass · ZeRO stages 1, 2, 3: shard optimizer state, gradients, weights; memory per GPU formula (widget `zero`) · FSDP: all-gather weights before each layer, reduce-scatter gradients after; communication volume vs plain data parallelism · the global batch size limit and why batch sizes are ramped up.

**7.6 Model parallelism (8).** Tensor parallelism: split the MLP's first matrix by columns and the second by rows, one all-reduce per block in each direction (widget `tp`) · tensor parallel attention by heads · sequence parallelism for norms and dropout · pipeline parallelism: GPipe schedule and its bubble (p − 1)/(m + p − 1) (widget `pp`) · 1F1B and interleaved schedules · expert parallelism and all-to-all for MoE · context parallelism for long sequences (ring attention) · putting it together: Llama 3.1 405B with TP 8 × PP 16 × DP 128 on 16,384 H100s (widget `mesh`) and how to choose the layout.

**7.7 Running a training job (6).** The data pipeline: crawl, filter, deduplicate, mix, tokenize, shard (widget `data`) · checkpointing: size, frequency, asynchronous and sharded writes · failures at scale (widget `failures`) · loss spikes and stability: warmup, gradient clipping, z-loss, skipping bad batches, restarting from a checkpoint · what to monitor: loss, gradient norm, tokens/s, MFU, per-GPU memory · evaluation during training.

Owns widgets: `flops`, `scaling`, `memory`, `precision`, `allreduce`, `zero`, `tp`, `pp`, `mesh`, `data`, `failures`.

---

## Part 8 · Inference and serving (≈ 38) · prefix `serve-` · file `content/08-serving.js`
Goal: how a request is served: prefill and decode, the KV cache, batching and scheduling, memory management, speculative decoding, quantization, parallelism for inference, and serving systems with capacity math.

**8.1 Two phases (5).** Prefill processes the prompt in parallel; decode produces one token per forward pass (widget `prefill`) · prefill is compute-bound, decode is memory-bound: bytes read per decoded token vs FLOPs (widget `membound`) · latency metrics: time to first token, time per output token, end-to-end, throughput; SLOs · the minimum decode step time for Llama 3.1 70B on 8 H100s from bandwidth alone · batching raises decode throughput almost for free until compute or memory runs out.

**8.2 The KV cache (6).** Keys and values of past tokens never change, so cache them (widget `kvcache`) · size per token = 2 × layers × KV heads × head dim × bytes; calculator (widget `kvsize`); Llama 3 70B at 128K context · GQA/MQA cut the cache (refer to Part 5) · the cache limits batch size: memory budget worked for an 80 GB GPU · prefix caching: reuse the cache of shared prompt prefixes (radix tree in SGLang) · KV cache offload and quantization.

**8.3 Batching and scheduling (6).** Static batching wastes slots on finished requests · continuous (iteration-level) batching (widget `batching`) · PagedAttention: fixed-size KV blocks and block tables; fragmentation before and after (widget `paged`) · chunked prefill to keep decode latency steady · preemption: swap or recompute when memory runs out · prefill/decode disaggregation (separate GPU pools).

**8.4 Making decode faster (8).** Speculative decoding: draft, then verify k tokens in one pass; expected tokens per pass (1 − α^(k+1))/(1 − α) (widget `speculative`) · draft models vs extra heads (Medusa, EAGLE) in one sheet · weight quantization: INT8, INT4, per-channel and group scales, error (widget `quant`) · GPTQ and AWQ, what each does · FP8 inference on Hopper · KV-cache quantization · FlashDecoding (split the KV sequence across blocks for long contexts) · CUDA graphs and small-batch launch overhead.

**8.5 Parallelism for inference (5).** Tensor parallelism to cut latency; the all-reduce cost per layer over NVLink · pipeline parallelism for throughput · expert parallelism for serving MoE models (DeepSeek-V3's deployment as published) · choosing GPUs per replica from memory and latency (calculator) · replicas and load balancing.

**8.6 Serving systems (8).** The request path: tokenize, schedule, prefill, decode, detokenize, stream · the architecture of vLLM (scheduler, block manager, workers) · constrained decoding: mask tokens that break a grammar; with our SQL grammar the model can never produce an invalid query (custom figure using `TinyLM`) · structured output (JSON schema) the same way · cost per million tokens from GPU price and throughput (calculator) · capacity planning for a target SLO · routing by prefix cache and fleet layout (widget `fleet`) · a consolidating sheet: one request's full timeline.

Owns widgets: `prefill`, `membound`, `kvcache`, `kvsize`, `batching`, `paged`, `speculative`, `quant`, `fleet`.

---

## Part 9 · Post-training (≈ 30) · prefix `post-` · file `content/09-post.js`
Goal: how a pretrained model becomes an assistant and a reasoner: SFT, LoRA, reward models, RLHF with PPO, DPO, RL with verifiable rewards (GRPO), the RL infrastructure, evaluation.

**9.1 From base model to assistant (4).** What a base model does with a question (continues the text) · chat templates and special tokens · the pipeline: SFT → preference optimization → RL, with data sizes · what each stage changes.

**9.2 Supervised fine-tuning (6).** Instruction–response data · the loss only on response tokens (widget `sft`) · packing sequences and attention masks · LoRA: W + BA with rank r; parameter count r(d + k) vs d·k, worked for d = k = 4,096, r = 16 · QLoRA (4-bit base + LoRA) · forgetting and data mixing.

**9.3 Preferences (7).** Pairwise human preference data · Bradley–Terry: p(A ≻ B) = σ(r_A − r_B), worked · training a reward model with that loss · RLHF with PPO: policy, reference, reward model, value model; the KL penalty; four models in memory (widget `prefs` on one of these sheets) · the PPO clipped objective, worked · DPO: the same goal without a reward model or sampling; derive the loss; the role of β · comparing PPO and DPO in practice.

**9.4 RL with verifiable rewards (6).** Rewards from checkers: unit tests, math answers, SQL execution (for our grammar: does the query parse, does it return the asked row) · policy gradient (REINFORCE) on a 3-action example with numbers · baselines and advantages reduce variance · GRPO: advantages relative to a group of samples, no value model (widget `rl`) · reasoning models and test-time compute: longer chains of thought, measured gains (cite) · reward hacking and how it shows up.

**9.5 RL infrastructure (4).** Generation dominates RL compute: an inference engine inside the training loop (widget `rlinfra`) · moving updated weights from trainers to generators · asynchronous, off-policy RL and staleness · scaling RL runs.

**9.6 Evaluation (3).** Benchmarks, contamination, and held-out tests · LLM-as-judge and its biases · a consolidating sheet: the full life of a model from random weights to deployed assistant.

Owns widgets: `sft`, `prefs`, `rl`, `rlinfra`.

---

## Part 10 · Wrap-up (6, written by the editor) · prefix `end-`
The whole journey of the query in one figure with links to sheets · what to study next and the papers to read · mapping sheets to interview topics · sources · how this set was made and verified.
