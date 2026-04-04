# main.py
from dotenv import load_dotenv
load_dotenv()

print("[INIT] Building graph... (TF + Phi-3 load here, expected)")
from graph import app
print("[INIT] Graph ready.\n")

def run(query: str):
    print(f"\n{'='*55}")
    print(f"  Query: {query}")
    print(f"{'='*55}")

    final_result = None

    # ── REMOVE app.invoke() — stream() alone gives you everything ────────────
    for step in app.stream({"query": query}, stream_mode="updates"):
        for node_name, output in step.items():
            print(f"\n┌── Node: [{node_name}]")
            for key, val in output.items():
                if hasattr(val, 'shape'):
                    print(f"│   {key}: shape={val.shape} → {val}")
                elif isinstance(val, dict):
                    print(f"│   {key}:")
                    for k, v in val.items():
                        print(f"│     {k}: {v}")
                else:
                    print(f"│   {key}: {val}")
            print(f"└{'─'*40}")

            if "final_response" in output:
                final_result = output["final_response"]

    print(f"\n{'='*55}")
    print(f"  Final Response:")
    print(f"{'='*55}")
    print(final_result)
    return final_result

if __name__ == "__main__":
    run('Explain  Rural Energy Access and Sustainability Initiatives?')