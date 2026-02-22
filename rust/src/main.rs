use memmap2::MmapOptions;
use std::fs::File;
use std::io::{self, Write};
use std::slice;
use std::time::Instant;
use tokenizers::Tokenizer;

// Holds the 'Config' of the hyperparameters of the nn
#[derive(Debug, Clone)]
struct Config {
    vocab_size: usize,
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
}

struct LayerWeights<'a> {
    attn_norm: &'a [f32],
    attn_post_norm: &'a [f32],
    // Attention
    wq: &'a [f32], // Queries
    wk: &'a [f32], // Keys
    wv: &'a [f32], // Values
    wo: &'a [f32], // Out

    ffn_pre_norm: &'a [f32],  // Norm before FFN
    ffn_post_norm: &'a [f32], // After FFN

    // FFN (SwiGLU)
    w1: &'a [f32], // Gate projection
    w2: &'a [f32], // Up projection
    w3: &'a [f32], // Down projection
}

struct Weights<'a> {
    token_embedding_table: &'a [f32],
    layers: Vec<LayerWeights<'a>>,
    final_norm: &'a [f32],
    wcls: &'a [f32],
}

struct RunState {
    x: Vec<f32>,      //
    xb: Vec<f32>,     // temp buff
    xb2: Vec<f32>,    // temp buff
    hb: Vec<f32>,     // buff FFN hidden state
    hb2: Vec<f32>,    // buff FFN hidden state
    q: Vec<f32>,      // Query
    k: Vec<f32>,      // Key
    v: Vec<f32>,      // Value
    att: Vec<f32>,    // Att Scores past and current token
    logits: Vec<f32>, // Raw probs

    // Kv-cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn new(c: &Config) -> Self {
        let kv_dim = c.n_kv_heads * c.head_dim;
        Self {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.n_heads * c.head_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; c.n_heads * c.max_seq_len],
            logits: vec![0.0; c.vocab_size],
            key_cache: vec![0.0; c.n_layers * c.max_seq_len * kv_dim],
            value_cache: vec![0.0; c.n_layers * c.max_seq_len * kv_dim],
        }
    }
}
// MATH

fn rmsnorm(o: &mut [f32], x: &[f32], w: &[f32]) {
    let size = x.len();
    // Calc the main of the squares
    let mut ss = x.iter().map(|v| v * v).sum::<f32>() / size as f32;
    ss = 1.0 / (ss + 1e-6).sqrt();
    for i in 0..size {
        // +1 important
        o[i] = (1.0 + w[i]) * (x[i] * ss);
    }
}

fn matmul(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    let d = out.len();
    for i in 0..d {
        let mut sum = 0.0;
        let row = &w[i * n..(i + 1) * n];
        for j in 0..n {
            sum += row[j] * x[j]
        }
        out[i] = sum;
    }
}

// Rope

fn apply_rope(v: &mut [f32], pos: usize, n_heads: usize, head_dim: usize) {
    let half_dim = head_dim / 2;
    for h in 0..n_heads {
        let head_offset = h * head_dim;
        for i in 0..half_dim {
            let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
            let (s, c) = ((pos as f32) * freq).sin_cos();

            let idx1 = head_offset + i;
            let idx2 = head_offset + i + half_dim;

            let v1 = v[idx1];
            let v2 = v[idx2];

            v[idx1] = v1 * c - v2 * s;
            v[idx2] = v2 * c + v1 * s;
        }
    }
}

fn forward(c: &Config, w: &Weights, s: &mut RunState, token: usize, pos: usize) {
    // EMBEDDING LOOKUP
    s.x.copy_from_slice(&w.token_embedding_table[token * c.dim..(token + 1) * c.dim]);

    // Scaling the initial embedding by the square root of the hidden dimension size
    let scale = (c.dim as f32).sqrt();
    for val in s.x.iter_mut() {
        *val *= scale;
    }

    // Pass vector to all 26 layers in sequence
    for l in 0..c.n_layers {
        let lw = &w.layers[l];

        // --- ATT ---

        // Normalize the vector
        rmsnorm(&mut s.xb, &s.x, lw.attn_norm);

        // Queries, Keys, and Values
        matmul(&mut s.q, &s.xb, lw.wq);
        matmul(&mut s.k, &s.xb, lw.wk);
        matmul(&mut s.v, &s.xb, lw.wv);

        // Apply Rope
        apply_rope(&mut s.q, pos, c.n_heads, c.head_dim);
        apply_rope(&mut s.k, pos, c.n_kv_heads, c.head_dim);

        // KV CACHE UPDATE
        let kv_dim = c.n_kv_heads * c.head_dim;
        let loff = l * c.max_seq_len * kv_dim;
        s.key_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim].copy_from_slice(&s.k);
        s.value_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim].copy_from_slice(&s.v);

        // GQA Attn
        let kv_mul = c.n_heads / c.n_kv_heads;
        let out_dim = c.n_heads * c.head_dim;

        for h in 0..c.n_heads {
            let q_off = h * c.head_dim;
            let kv_off = (h / kv_mul) * c.head_dim;

            for t in 0..=pos {
                let mut score = 0.0;
                let k_base = loff + t * kv_dim + kv_off;

                for i in 0..c.head_dim {
                    score += s.q[q_off + i] * s.key_cache[k_base + i];
                }

                score /= (c.head_dim as f32).sqrt();

                // Attn soft cap -50.0 and +50.0 with scaled Tanh func
                s.att[h * c.max_seq_len + t] = 50.0 * (score / 50.0).tanh();
            }

            // Softmax
            let att = &mut s.att[h * c.max_seq_len..h * c.max_seq_len + pos + 1];
            let max_a = att.iter().cloned().fold(f32::NEG_INFINITY, f32::max); // Find max for numerical stability
            let mut sum_a = 0.0;
            for v in att.iter_mut() {
                *v = (*v - max_a).exp();
                sum_a += *v;
            }
            for v in att.iter_mut() {
                *v /= sum_a;
            }

            // Weighted Sum of Values
            for i in 0..c.head_dim {
                let mut val = 0.0;
                for t in 0..=pos {
                    val += att[t] * s.value_cache[loff + t * kv_dim + kv_off + i];
                }
                s.xb2[q_off + i] = val; // Store the result in xb2
            }
        }

        // Wo
        matmul(&mut s.xb, &s.xb2[..out_dim], lw.wo);

        // Post-Attn norm
        rmsnorm(&mut s.xb2, &s.xb, lw.attn_post_norm);

        // Residual connection add to main vector `s.x`
        for i in 0..c.dim {
            s.x[i] += s.xb2[i];
        }

        // --- FFN ---

        // Normalize before the FFN
        rmsnorm(&mut s.xb, &s.x, lw.ffn_pre_norm);

        // Project into a larger dimension size (from 2304 up to 9216 in Gemma 2B)
        matmul(&mut s.hb, &s.xb, lw.w1); // The "Gate" layer
        matmul(&mut s.hb2, &s.xb, lw.w3); // The "Up" layer

        // Activation func SwiGLU
        for i in 0..c.hidden_dim {
            let v = s.hb[i];
            // Some math wizards found this approximation to the SwiGLU function that is much faster to compute.
            s.hb[i] = (0.5 * v * (1.0 + (0.79788 * (v + 0.044715 * v * v * v)).tanh())) * s.hb2[i];
        }

        // Project back down from 9216 down to 2304
        matmul(&mut s.xb2, &s.hb, lw.w2);

        //Post-FFN Norm.
        rmsnorm(&mut s.xb, &s.xb2, lw.ffn_post_norm);

        // Residual 2
        for i in 0..c.dim {
            s.x[i] += s.xb[i];
        }
    }

    // Final normalization
    rmsnorm(&mut s.xb, &s.x, w.final_norm);

    // Classifier: Multiply by the `wcls` matrix to get a score for all 256,000 words in the vocabulary.
    matmul(&mut s.logits, &s.xb, w.wcls);

    // Logits soft-capp  (-30.0 to +30.0)
    for v in s.logits.iter_mut() {
        *v = 30.0 * (*v / 30.0).tanh();
    }
}

fn main() {
    let tokenizer =
        Tokenizer::from_file("../tokenizer.json").expect("Failed to load tokenizer.json");

    let file = File::open("../gemma2_2b.bin").expect("Failed to open weights file");

    // MEMORY MAPPING: Kinda like magic. Rust to hard for me tech bros.
    let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to mmap file") };

    // We cast the first 8 chunks of 4 bytes into an array of eight `u32` integers.
    // Another way of saying more magic ...
    let h = unsafe { slice::from_raw_parts(mmap.as_ptr() as *const u32, 8) };
    let c = Config {
        vocab_size: h[0] as usize,
        dim: h[1] as usize,
        hidden_dim: h[2] as usize,
        n_layers: h[3] as usize,
        n_heads: h[4] as usize,
        n_kv_heads: h[5] as usize,
        max_seq_len: h[6] as usize,
        head_dim: h[7] as usize,
    };

    if c.vocab_size == 0 {
        panic!("Header parsing failed. Export script missing 'IIIIIIII' packing. Go learn python before trying to understand this Rust code.");
    }

    // MAP THE WEIGHTS
    // Skip the header(32 bytes) and interpret the rest as f32 weights
    let d = unsafe {
        slice::from_raw_parts(mmap.as_ptr().add(32) as *const f32, (mmap.len() - 32) / 4)
    };

    let mut off = 0;

    let mut g = |s: usize| {
        let r = &d[off..off + s];
        off += s;
        r
    };

    // Populate the weights struct by chopping up the massive memory mapped array in the exact order
    let mut w = Weights {
        token_embedding_table: g(c.vocab_size * c.dim),
        layers: vec![],
        final_norm: &[],
        wcls: &[],
    };

    for _ in 0..c.n_layers {
        w.layers.push(LayerWeights {
            attn_norm: g(c.dim),
            attn_post_norm: g(c.dim),
            wq: g(c.dim * c.n_heads * c.head_dim),
            wk: g(c.dim * c.n_kv_heads * c.head_dim),
            wv: g(c.dim * c.n_kv_heads * c.head_dim),
            wo: g(c.n_heads * c.head_dim * c.dim),
            ffn_pre_norm: g(c.dim),
            ffn_post_norm: g(c.dim),

            w1: g(c.dim * c.hidden_dim), // gate
            w3: g(c.dim * c.hidden_dim), // up
            w2: g(c.hidden_dim * c.dim), // down
        });
    }
    w.final_norm = g(c.dim);
    w.wcls = g(c.vocab_size * c.dim);

    // Pre-allocate all memory buffers for runtime calculations.
    let mut s = RunState::new(&c);

    // Format the prompt using Gemma's specific instruct template.
    let prompt = "<bos><start_of_turn>user\nWhat is the capital of Portugal ?<end_of_turn>\n<start_of_turn>model\n";

    // Encode the prompt into integers false because <bos> is already in the prompt
    let tokens: Vec<usize> = tokenizer
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&i| i as usize)
        .collect();

    let mut pos = 0;
    print!("Prefilling prompt");
    io::stdout().flush().unwrap();

    //PREFILL
    for i in 0..tokens.len() - 1 {
        forward(&c, &w, &mut s, tokens[i], pos);
        pos += 1;
        print!(".");
        io::stdout().flush().unwrap();
    }
    println!("\n\n--- Generating ---");

    // Next token starts as the last token of the prompt
    let mut next = tokens.last().copied().unwrap();
    let start = Instant::now();
    let mut gen_count = 0;

    // GENERATION LOOP
    for _ in 0..100 {
        //up to 100 tokens

        forward(&c, &w, &mut s, next, pos);

        // GREEDY DECODING: Just pick the token with highest logit score, kinda works tho
        next = s
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        pos += 1;
        gen_count += 1;

        // STOPPING CRITERIA
        // Token 1 is `<eos>` (End of Sequence).
        // Token 107 is `<end_of_turn>` in Gemma.
        if next == 1 || next == 107 {
            break;
        }

        let piece = tokenizer.decode(&[next as u32], false).unwrap_or_default();
        print!("{}", piece);
        io::stdout().flush().unwrap();
    }

    // Performance tracking to see how badly optimized this hot mess of a Rust code is sorry
    println!(
        "\n\nSpeed: {:.2} tok/s",
        gen_count as f32 / start.elapsed().as_secs_f32()
    );
}
