import memfiles, math, times, strformat, os, osproc, json, strutils
import nimpy

type LlmTokenizer = object
  pyTok: PyObject

let pyTokenizers = pyImport("tokenizers")

proc loadTokenizer(path: string): LlmTokenizer =
  echo "Load Tokenizer"


proc encode(t: LlmTokenizer, text: string, add_special: bool): seq[int] =
  let encoded = t.pyTok.encode(text, add_special_tokens = add_special)
  result = encoded.ids.to(seq[int])

proc decode(t: LlmTokenizer, id: int): string =
  result = t.pyTok.decode([id]).to(string)

const text: string = "hello"

const encoded: seq[int] = encode(pyTokenizers, text, nil)

echo text
