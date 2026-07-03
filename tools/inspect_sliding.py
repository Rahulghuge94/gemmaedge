from gguf import GGUFReader

reader = GGUFReader("/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/02749a7b272109255a4c559a80894d3d9777574c/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")

for key in ["gemma-4-26b-A4B-it-qat-UD-Q4_K_XL", "gemma4.attention.sliding_window_pattern", "gemma4.attention.head_count_kv"]:
    field = reader.fields.get(key)
    if field:
        print(f"Key: {key}")
        arr = field.parts[-1]
        print("  Type of parts[-1]:", type(arr))
        try:
            print("  Length/Shape:", len(arr) if hasattr(arr, '__len__') else getattr(arr, 'shape', 'No shape'))
            # Print elements
            elements = []
            for i in range(len(arr)):
                elements.append(str(arr[i]))
            print("  Elements:", ", ".join(elements))
        except Exception as e:
            print("  Error printing elements:", e)
