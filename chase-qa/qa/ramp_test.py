"""Find ELM's safe concurrency level.

Fires small identical requests at increasing concurrency and reports success rate,
latency and throughput at each level. Deliberately does NOT retry: the point is to
see failures, not to paper over them.

    python ramp_test.py -model gpt-5.4-mini
    python ramp_test.py -model gpt-5.4-mini -levels 1,4,8,16,32,48

Read the output like this:
  - throughput climbing, errors zero      -> raise concurrency
  - throughput flat, latency climbing     -> saturated, back off one level
  - any 429 / rate limit errors           -> hard ceiling, stay below it
"""

import os
import sys
import time
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def build_parser():
	p = argparse.ArgumentParser(description='ELM concurrency ramp test')
	p.add_argument('-model', type=str, required=True, help='Model string, e.g. gpt-5.4-mini')
	p.add_argument('-levels', type=str, default='1,2,4,8,16,32', help='Comma-separated concurrency levels')
	p.add_argument('-max_tokens', type=int, default=8, help='Response cap, keep small')
	p.add_argument('-cooldown', type=float, default=3.0, help='Seconds to wait between levels')
	return p


def one_call(client, model, max_tokens):
	"""Returns (ok, seconds, error_label)."""
	t0 = time.time()
	try:
		kwargs = {
			"model": model,
			"messages": [{"role": "user", "content": "Reply with the single word: ok"}],
		}
		# Mirrors models.py: gpt-5* uses max_completion_tokens and rejects `stop`.
		if model.startswith("gpt-5"):
			kwargs["max_completion_tokens"] = max_tokens
		else:
			kwargs["max_tokens"] = max_tokens
		client.chat.completions.create(**kwargs)
		return True, time.time() - t0, None
	except Exception as e:
		label = type(e).__name__
		text = str(e).lower()
		if "429" in text or ("rate" in text and "limit" in text):
			label = "RATE_LIMIT"
		elif "timeout" in text:
			label = "TIMEOUT"
		elif "connection" in text:
			label = "CONNECTION"
		return False, time.time() - t0, label


def run_level(client, model, n, max_tokens):
	t0 = time.time()
	with ThreadPoolExecutor(max_workers=n) as ex:
		results = list(ex.map(lambda _: one_call(client, model, max_tokens), range(n)))
	wall = time.time() - t0

	oks = [r for r in results if r[0]]
	errs = [r for r in results if not r[0]]
	lats = sorted(r[1] for r in oks)

	def pct(p):
		if not lats:
			return float('nan')
		return lats[min(len(lats) - 1, int(len(lats) * p))]

	err_counts = {}
	for _, _, label in errs:
		err_counts[label] = err_counts.get(label, 0) + 1

	return {
		"n": n,
		"ok": len(oks),
		"fail": len(errs),
		"wall": wall,
		"p50": statistics.median(lats) if lats else float('nan'),
		"p95": pct(0.95),
		"throughput": len(oks) / wall if wall > 0 else 0.0,
		"errors": err_counts,
	}


def main(args):
	key = os.environ.get("ELM_API_KEY")
	base = os.environ.get("ELM_BASE_URL")
	if not key or not base:
		print("ERROR: ELM_API_KEY and ELM_BASE_URL must be set.")
		sys.exit(1)

	client = OpenAI(api_key=key, base_url=base)
	levels = [int(x) for x in args.levels.split(",") if x.strip()]

	print("Model: {}   Base: {}".format(args.model, base))
	print("Levels: {}   Total calls: {}".format(levels, sum(levels)))
	print()
	print("{:>5} {:>5} {:>5} {:>8} {:>8} {:>8} {:>10}  {}".format(
		"conc", "ok", "fail", "wall_s", "p50_s", "p95_s", "calls/s", "errors"))
	print("-" * 78)

	rows = []
	for n in levels:
		r = run_level(client, args.model, n, args.max_tokens)
		rows.append(r)
		err_str = ", ".join("{}x{}".format(v, k) for k, v in sorted(r["errors"].items())) or "-"
		print("{:>5} {:>5} {:>5} {:>8.2f} {:>8.2f} {:>8.2f} {:>10.2f}  {}".format(
			r["n"], r["ok"], r["fail"], r["wall"], r["p50"], r["p95"], r["throughput"], err_str))
		if r["fail"] == r["n"]:
			print("\nAll calls failed at concurrency {}. Stopping.".format(n))
			break
		time.sleep(args.cooldown)

	print()
	clean = [r for r in rows if r["fail"] == 0]
	if not clean:
		print("No level completed without errors. Investigate the errors above before sharding.")
		return

	best = max(clean, key=lambda r: r["throughput"])
	print("Best error-free throughput: {:.2f} calls/s at concurrency {}".format(best["throughput"], best["n"]))

	rate_limited = [r["n"] for r in rows if "RATE_LIMIT" in r["errors"]]
	if rate_limited:
		print("Rate limiting first seen at concurrency {}. Keep total shards below that.".format(min(rate_limited)))
	else:
		print("No rate limiting observed up to concurrency {}.".format(max(r["n"] for r in rows)))

	print()
	single = rows[0]["p50"] if rows else float('nan')
	print("Suggested shard count: {} (leaves headroom; each shard is one sequential process).".format(
		max(1, int(best["n"] * 0.75))))
	print("Single-call p50 was {:.2f}s. Use this to sanity check the runtime estimates.".format(single))


if __name__ == "__main__":
	main(build_parser().parse_args())
