"""Measure real throughput and find the safe shard count.

Two things matter for sizing shards, and they are different:
  1. Request concurrency  - how many calls in flight are tolerated (RPM)
  2. Token throughput     - how many output tokens per minute are tolerated (TPM)

Document generation emits ~2000 output tokens per call, so TPM binds long before
RPM does. A ramp using one-word replies measures the wrong limit.

This script reports the account's advertised limits from response headers, then
ramps with realistic payloads and measures actual tokens/second.

    python throughput_test.py -model gpt-5.4-mini                 # headers + short + long
    python throughput_test.py -model gpt-5.4-mini -profile long   # long only
    python throughput_test.py -model gpt-5.4-mini -levels 1,4,8   # cheaper

COST: the long profile generates real output. Default levels total 29 calls at up
to 2000 output tokens each, roughly 58k output tokens. Reduce -levels to spend less.
"""

import os
import sys
import time
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

SHORT_PROMPT = "Reply with the single word: ok"

# Mirrors the shape of programmatic_docs generation: long transcript, high token count.
LONG_PROMPT = (
	"Write a detailed transcript of a meeting between a UK financial adviser and their client "
	"discussing pension consolidation. Begin with a line 'Meeting: <topic>', then dialogue where "
	"every line starts with 'Adviser:' or 'Client:'. Include specific figures, named rules and "
	"product details throughout. Write at least 1200 words. Do not summarise or conclude."
)


def build_parser():
	p = argparse.ArgumentParser(description='Throughput and concurrency test')
	p.add_argument('-model', type=str, required=True)
	p.add_argument('-profile', type=str, default='both', choices=['short', 'long', 'both'])
	p.add_argument('-levels', type=str, default='1,4,8,16', help='Comma-separated concurrency levels')
	p.add_argument('-long_max_tokens', type=int, default=2000, help='Output cap for the long profile')
	p.add_argument('-cooldown', type=float, default=5.0, help='Seconds between levels')
	return p


def make_kwargs(model, prompt, max_tokens):
	kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
	# Mirrors models.py: gpt-5* uses max_completion_tokens and rejects `stop`.
	if model.startswith("gpt-5"):
		kw["max_completion_tokens"] = max_tokens
	else:
		kw["max_tokens"] = max_tokens
	return kw


def show_limits(client, model):
	"""Definitive limits, straight from the response headers."""
	print("=== Advertised account limits ===")
	try:
		raw = client.chat.completions.with_raw_response.create(**make_kwargs(model, SHORT_PROMPT, 8))
	except Exception as e:
		print("  Could not read headers: {}: {}".format(type(e).__name__, e))
		print()
		return
	interesting = [
		"x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
		"x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
	]
	found = False
	for h in interesting:
		v = raw.headers.get(h)
		if v is not None:
			found = True
			print("  {:<34} {}".format(h, v))
	if not found:
		print("  No x-ratelimit-* headers returned.")
	print()


def one_call(client, model, prompt, max_tokens):
	"""Returns (ok, seconds, completion_tokens, error_label)."""
	t0 = time.time()
	try:
		r = client.chat.completions.create(**make_kwargs(model, prompt, max_tokens))
		dt = time.time() - t0
		out = 0
		if getattr(r, "usage", None) is not None:
			out = getattr(r.usage, "completion_tokens", 0) or 0
		return True, dt, out, None
	except Exception as e:
		dt = time.time() - t0
		text = str(e).lower()
		label = type(e).__name__
		if "429" in text or ("rate" in text and "limit" in text):
			label = "TPM/RPM_LIMIT" if "token" in text else "RATE_LIMIT"
		elif "timeout" in text:
			label = "TIMEOUT"
		elif "connection" in text:
			label = "CONNECTION"
		return False, dt, 0, label


def run_level(client, model, n, prompt, max_tokens):
	t0 = time.time()
	with ThreadPoolExecutor(max_workers=n) as ex:
		res = list(ex.map(lambda _: one_call(client, model, prompt, max_tokens), range(n)))
	wall = time.time() - t0

	oks = [r for r in res if r[0]]
	errs = [r for r in res if not r[0]]
	lats = sorted(r[1] for r in oks)
	toks = sum(r[2] for r in oks)

	counts = {}
	for _, _, _, label in errs:
		counts[label] = counts.get(label, 0) + 1

	return {
		"n": n, "ok": len(oks), "fail": len(errs), "wall": wall,
		"p50": statistics.median(lats) if lats else float('nan'),
		"p95": lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else float('nan'),
		"tokens": toks,
		"calls_s": len(oks) / wall if wall > 0 else 0.0,
		"tok_s": toks / wall if wall > 0 else 0.0,
		"errors": counts,
	}


def ramp(client, model, levels, prompt, max_tokens, cooldown, label):
	print("=== {} profile (cap {} output tokens) ===".format(label, max_tokens))
	print("{:>5} {:>4} {:>5} {:>8} {:>7} {:>7} {:>9} {:>9} {:>9}  {}".format(
		"conc", "ok", "fail", "wall_s", "p50_s", "p95_s", "out_toks", "calls/s", "toks/s", "errors"))
	print("-" * 92)
	rows = []
	for n in levels:
		r = run_level(client, model, n, prompt, max_tokens)
		rows.append(r)
		err = ", ".join("{}x{}".format(v, k) for k, v in sorted(r["errors"].items())) or "-"
		print("{:>5} {:>4} {:>5} {:>8.1f} {:>7.1f} {:>7.1f} {:>9} {:>9.2f} {:>9.0f}  {}".format(
			r["n"], r["ok"], r["fail"], r["wall"], r["p50"], r["p95"], r["tokens"],
			r["calls_s"], r["tok_s"], err))
		if r["fail"] == r["n"]:
			print("\nAll calls failed at concurrency {}. Stopping this profile.".format(n))
			break
		time.sleep(cooldown)
	print()
	return rows


def summarise(rows, label):
	clean = [r for r in rows if r["fail"] == 0]
	if not clean:
		print("{}: no error-free level. Investigate errors before sharding.".format(label))
		return None
	best = max(clean, key=lambda r: r["tok_s"] if r["tokens"] else r["calls_s"])
	print("{}: best error-free level = {} ({:.2f} calls/s, {:.0f} output tokens/s)".format(
		label, best["n"], best["calls_s"], best["tok_s"]))
	limited = [r["n"] for r in rows if r["errors"]]
	if limited:
		print("{}: first errors at concurrency {}".format(label, min(limited)))
	# Saturation: throughput gain under 20% while concurrency doubled
	for a, b in zip(clean, clean[1:]):
		if b["n"] >= 2 * a["n"] and a["tok_s"] > 0 and b["tok_s"] < a["tok_s"] * 1.2:
			print("{}: throughput flattened between {} and {} -> saturated around {}".format(
				label, a["n"], b["n"], a["n"]))
			break
	return best


def main(args):
	key = os.environ.get("ELM_API_KEY")
	base = os.environ.get("ELM_BASE_URL")
	if not key or not base:
		print("ERROR: ELM_API_KEY and ELM_BASE_URL must be set.")
		sys.exit(1)

	client = OpenAI(api_key=key, base_url=base)
	levels = [int(x) for x in args.levels.split(",") if x.strip()]

	print("Model: {}   Base: {}   Levels: {}".format(args.model, base, levels))
	print()

	show_limits(client, args.model)

	short_best = long_best = None
	if args.profile in ("short", "both"):
		rows = ramp(client, args.model, levels, SHORT_PROMPT, 8, args.cooldown, "SHORT")
		short_best = summarise(rows, "SHORT")
		print()

	if args.profile in ("long", "both"):
		rows = ramp(client, args.model, levels, LONG_PROMPT, args.long_max_tokens, args.cooldown, "LONG")
		long_best = summarise(rows, "LONG")
		print()

	binding = long_best or short_best
	if binding:
		print("=== Sizing ===")
		print("Binding profile concurrency: {}".format(binding["n"]))
		print("Suggested shard count: {} (75% headroom)".format(max(1, int(binding["n"] * 0.75))))
		if long_best and long_best["p50"] == long_best["p50"]:
			print("Long-call p50 {:.1f}s. Doc generation dominates runtime; "
			      "one shard's wall-clock is roughly (its rows x 4 x p50) for stage 7.".format(long_best["p50"]))


if __name__ == "__main__":
	main(build_parser().parse_args())
