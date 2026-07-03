from gguf import GGUFReader

reader = GGUFReader("/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/02749a7b272109255a4c559a80894d3d9777574c/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
field = reader.fields.get("tokenizer.chat_template")
if field:
    val = field.parts[-1]
    if not isinstance(val, (str, bytes)):
        # Convert numpy uint8 array to bytes
        val = bytes(val)
    if isinstance(val, bytes):
        val = val.decode('utf-8', errors='replace')
    print("Chat Template:")
    print(val)
else:
    print("Chat Template not found.")
