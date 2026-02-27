import memfiles, math, times, strformat, os, osproc, json, strutils
import nimpy

type LlmTokenizer = object
  pyTok: PyObject

let pyTokenizers = pyImport("tokenizers")

proc loadTokenizer(path: string): LlmTokenizer =
  echo "Load Tokenizer"
  result.pyTok = pyTokenizers.Tokenizer.from_file(path)


proc encode(t: LlmTokenizer, text: string, add_special: bool): seq[int] =
  let encoded = t.pyTok.encode(text, add_special_tokens = add_special)
  result = encoded.ids.to(seq[int])

proc decode(t: LlmTokenizer, ids: seq[int]): string =
  result = t.pyTok.decode(ids).to(string)

let tokenizer = loadTokenizer("../tokenizer.json")

type Config = object
  vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads, max_seq_len, head_dim: int

type FloatArray = ptr UncheckedArray[float32]

type LayerWeights = object
  attn_norm, attn_post_norm, wq, wk, wv, wo, ffn_pre_norm, ffn_post_norm, w1,
    w3, w2: FloatArray

type Weights = object
  token_embedding_table: FloatArray
  layers: seq[LayerWeights]
  final_norm, wcls: FloatArray

type RunState = ref object
  x, xb, xb2, hb, hb2, q, k, v, att, logits, key_cache, value_cache: seq[float32]


proc newRunState(c: Config): RunState =
  let kv_dim = c.n_kv_heads * c.head_dim
  new(result)
  result.x = newSeq[float32](c.dim)
  result.xb = newSeq[float32](c.dim)
  result.xb2 = newSeq[float32](c.dim)
  result.hb = newSeq[float32](c.hidden_dim)
  result.hb2 = newSeq[float32](c.hidden_dim)
  result.q = newSeq[float32](c.n_heads * c.head_dim)
  result.k = newSeq[float32](kv_dim)
  result.v = newSeq[float32](kv_dim)
  result.att = newSeq[float32](c.n_heads * c.max_seq_len)
  result.logits = newSeq[float32](c.vocab_size)
  result.key_cache = newSeq[float32](c.n_layers * c.max_seq_len * kv_dim)
  result.value_cache = newSeq[float32](c.n_layers * c.max_seq_len * kv_dim)


# ============================================================================
# MATH the easy stuff
# ===========================================================================


proc rmsnorm(o: var seq[float32], x: openArray[float32], w: FloatArray) =
  var ss = 0.0'f32
  for i in 0 ..< x.len: ss += x[i] * x[i]
  ss = 1.0'f32 / sqrt((ss / x.len.float32) + 1e-6'f32)
  for i in 0 ..< x.len: o[i] = (1.0'f32 + w[i]) * (x[i] * ss)

proc matmul(out_v: var seq[float32], x: openArray[float32], w: FloatArray) =
  let n = x.len
  for i in 0 ..< out_v.len:
   var sum = 0.0'f32
   var off = i * n
   for j in 0 ..< n: sum += w[off + j] * x[j]
   out_v[i] = sum
   
proc apply_rope(v: var seq[float32],pos,n_heads, head_dim:int) =
  let half = head_dim div 2
  let freq = 1 / pow(10000.0, (2*i).float32 / head_dim.float32)
  for h in 0 ..< n_heads:

    let off = h * head_dim
    for i in 0 ..< half:
      let (s, c) = (sin(pos.float32 * freq), cos(pos.float32 * freq))
      let (v1, v2) =  = (v[off + i] v[off + i + half])
      v[off + i ]

