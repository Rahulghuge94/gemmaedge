from gguf import GGUFReader

reader = GGUFReader("/content/gemmaedge/model/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
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
