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
ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
args = ap.parse_args()

dev = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(args.dtype) or (torch.bfloat16 if dev == "cuda" else torch.float32)
print(f"loading {args.model} on {dev} ({dtype}) ...", flush=True)

# Official loading code from the Apertus-v1.5 model card (requires the swiss-ai transformers branch):
#   pip install "transformers[torch,vision,audio] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"
proc = transformers.AutoProcessor.from_pretrained(args.model)
tok = getattr(proc, "tokenizer", proc)
_cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
model = _cls.from_pretrained(args.model, dtype=dtype)
model.to(dev).eval()
LOCK = threading.Lock()
print(f"ready on http://localhost:{args.port}/v1", flush=True)


def generate(messages, max_tokens=512, temperature=0.3):
    for m in messages:  # flatten content blocks
        if isinstance(m.get("content"), list):
            m["content"] = " ".join(b.get("text", "") for b in m["content"] if isinstance(b, dict))
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(dev)
    ids = inputs["input_ids"]
    kw = dict(max_new_tokens=int(max_tokens), pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kw.update(do_sample=False)
    with LOCK, torch.inference_mode():
        out = model.generate(**inputs, **kw)
    return proc.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1], out.shape[1] - ids.shape[1]


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
