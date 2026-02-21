import struct
import torch
from transformers import AutoModelForCausalLM

model_id = "google/gemma-2-2b-it"
print(f"Loading {model_id}... This might take a minute.")

# Load model on cpu
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
config = model.config

vocab_size = config.vocab_size  # 256000
dim = config.hidden_size  # 2304
hidden_dim = config.intermediate_size  # 9216
n_layers = config.num_hidden_layers  # 26
n_heads = config.num_attention_heads  # 8
n_kv_heads = config.num_key_value_heads  # 4
max_seq_len = config.max_position_embeddings  # 8192
head_dim = config.head_dim  # 256

with open("gemma2_b.bin", "wb") as f:
    header = struct.pack(
        "iiiiiiii",
        vocab_size,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        max_seq_len,
        head_dim,
    )
    f.write(header)
    state_dict = model.state_dict()

    def write_tensor(tensor_name):
        if tensor_name not in state_dict:

            print(f"Export failt tensor: {tensor_name} not found.")
            return

        tensor = state_dict[tensor_name].float().flatten().numpy()
        f.write(tensor.tobytes())

    write_tensor("model.embed_tokens.weight")

    for i in range(n_layers):
        # Attention Norms
        write_tensor(f"model.layers.{i}.input_layernorm.weight")
        write_tensor(f"model.layers.{i}.post_attention_layernorm.weight")

        # Attention Projections (GQA)
        write_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
        write_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
        write_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
        write_tensor(f"model.layers.{i}.self_attn.o_proj.weight")

        # FFN Norms
        write_tensor(f"model.layers.{i}.pre_feedforward_layernorm.weight")
        write_tensor(f"model.layers.{i}.post_feedforward_layernorm.weight")

        # FFN Projections
        write_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
        write_tensor(f"model.layers.{i}.mlp.up_proj.weight")
        write_tensor(f"model.layers.{i}.mlp.down_proj.weight")

    # Final Norm & Output Head
    write_tensor("model.norm.weight")
    write_tensor("lm_head.weight")
print("Export complete!!")
