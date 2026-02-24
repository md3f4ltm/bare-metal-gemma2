# Bare-Metal Gemma 2

![](assets/gemma2.gif)

This is an implementation of the Gemma 2 (2B) inference engine in **Rust**. I might provide **C**, **Zig**, or **Nim** implementations in the future for fun.

I tried to make the code as simple and readable as possible, but my Rust skills are not perfect. I ran into too many issues with the borrow checker and ended up using `unsafe` to map the weight tensors directly to memory. If you have suggestions for doing this without `unsafe`, please open an issue or submit a pull request.

This project was inspired by Karpathy's llama2.c. He showed that the math behind transformer inference can be implemented without heavy frameworks — newer models are a bit more complex but still quite doable. You can check llama2.c here: https://github.com/karpathy/llama2.c

## Feel the power of 1 token/s

To try it out, clone the repo and run the following commands:

```bash
git clone https://github.com/md3f4ltm/bare-metal-gemma2.git
cd bare-metal-gemma2
```

Download and export the weights from Hugging Face using the `export.py` script. You will need to create a Hugging Face account and accept the model license to download the weights: https://huggingface.co/google/gemma-2-2b-it

```bash
python export.py
```

If you don't want to use the `export.py` script, i also uploaded the exported weights to hugging face.

```bash
wget https://huggingface.co/d3falt-dev/gemma2-2b-bin/resolve/main/gemma2_2b.bin
```

Compile the Rust code and run the inference:

```bash
cargo run --release
```

I include a language-agnostic mathematical breakdown of the model (written by an LLM) in the repository.

- Math breakdown [here](math.md).

## References
- [Google Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118)
- [Hugging Face Gemma 2 Modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py)
- [Rotary Embeddings (RoPE) Explained](https://blog.eleuther.ai/rotary-embeddings/)

## Dependencies

### Rust
- `memmap2` — for memory-mapping the weights file.
- `tokenizers` — for tokenization (it's possible to implement your own; an inefficient one is not too hard to do).

---

Have fun!
