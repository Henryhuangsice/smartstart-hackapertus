"""Minimal OpenAI-compatible server for Apertus with Hugging Face transformers.

Fallback when mlx-lm cannot run the model. Works on Apple Silicon (MPS),
NVIDIA (CUDA) or CPU.

    pip install torch transformers accelerate
    python scripts/serve_transformers.py --model swiss-ai/Apertus-v1.5-8B --port 8080
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import transformers
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev != "cpu" else torch.float32
print(f"loading {args.model} on {dev} ({dtype}) ...", flush=True)

tok = None
model = None
try:
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
except Exception as e:  # multimodal checkpoints
    print("AutoModelForCausalLM failed:", e, "\n-> trying AutoModelForImageTextToText", flush=True)
    proc = transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", proc)
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
model.to(dev).eval()
LOCK = threading.Lock()
print(f"ready on http://localhost:{args.port}/v1", flush=True)


def generate(messages, max_tokens=512, temperature=0.3):
    for m in messages:  # flatten content blocks
        if isinstance(m.get("content"), list):
            m["content"] = " ".join(b.get("text", "") for b in m["content"] if isinstance(b, dict))
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = ids.to(dev)
    kw = dict(max_new_tokens=int(max_tokens), pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kw.update(do_sample=False)
    with LOCK, torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1], out.shape[1] - ids.shape[1]


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": args.model, "object": "model"}]})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        t0 = time.time()
        try:
            text, n_in, n_out = generate(req.get("messages", []), req.get("max_tokens", 512), req.get("temperature", 0.3))
        except Exception as e:
            return self._send(500, {"error": str(e)})
        print(f"[{n_in}->{n_out} tok, {time.time()-t0:.1f}s]", flush=True)
        self._send(200, {"id": "chatcmpl", "object": "chat.completion", "model": args.model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": n_in, "completion_tokens": n_out}})

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
