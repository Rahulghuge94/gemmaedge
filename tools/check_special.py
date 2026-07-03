from gguf import GGUFReader

reader = GGUFReader("/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/02749a7b272109255a4c559a80894d3d9777574c/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
field = reader.fields.get("tokenizer.ggml.tokens")
if field:
    print("Field type:", type(field))
    # print all fields/attributes of field
    print("Attrs:", dir(field))
    # If it is an array of strings, how do we access it?
    # GGUF fields usually have a list of parts, or field.parts contains numpy array if numeric,
    # or list of bytes if string array.
    # Let's print the length of parts and their types
    print("Parts count:", len(field.parts))
    for i, p in enumerate(field.parts):
        print(f"Part {i} type: {type(p)}")
        if isinstance(p, (list, tuple)):
            print(f"Part {i} length: {len(p)}")
            if len(p) > 0:
                print(f"Part {i} first element type: {type(p[0])}")
        else:
            try:
                print(f"Part {i} shape: {p.shape}, dtype: {p.dtype}")
            except Exception:
                pass
