import sys
from gguf import GGUFReader

if len(sys.argv) < 2:
    print("usage: python inspect_tokens.py MODEL.gguf")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])
tokens_key = "tokenizer.ggml.tokens"
field = reader.get_field(tokens_key)
if field is None:
    print("Tokenizer tokens not found in metadata.")
    sys.exit(1)

tokens = field.parts[-1] # List of bytes
target_ids = [236779, 236794, 1340, 45518, 100]
for tid in target_ids:
    if tid < len(tokens):
        val = tokens[tid]
        if isinstance(val, bytes):
            # Print only ascii characters, replace others with hex
            ascii_chars = []
            for b in val:
                if 32 <= b <= 126:
                    ascii_chars.append(chr(b))
                else:
                    ascii_chars.append(f"\\x{b:02x}")
            decoded = "".join(ascii_chars)
        else:
            decoded = str(val)
        print(f"Token {tid}: {decoded}")
    else:
        print(f"Token {tid}: out of range")
