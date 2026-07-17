import json
import random

def load_generic_seeds(path="generic_seeds.json"):
	with open(path, "r") as f:
		return json.load(f)

def sample_value(spec):
	if spec["type"] == "int":
		return random.randint(spec["range"][0], spec["range"][1])
	elif spec["type"] == "float":
		return round(random.uniform(spec["range"][0], spec["range"][1]), 4)
	else:
		raise ValueError(f"Unknown variable type: {spec['type']}")

def evaluate_formula(formula_str, sampled_vars):
	namespace = dict(sampled_vars)
	statements = [s.strip() for s in formula_str.split(";") if s.strip()]
	for stmt in statements:
		exec(stmt, {}, namespace)
	return namespace

def build_numerics_text(sampled_vars, computed_vars, formula_str):
	assigned_names = set()
	for stmt in formula_str.split(";"):
		stmt = stmt.strip()
		if "=" in stmt:
			assigned_names.add(stmt.split("=")[0].strip())

	lines = ["Sampled inputs:"]
	for k, v in sampled_vars.items():
		lines.append(f"- {k}: {v}")
	lines.append("")
	lines.append("Computed values:")
	for k in assigned_names:
		if k in computed_vars:
			lines.append(f"- {k}: {computed_vars[k]}")
	return "\n".join(lines)

def get_sampled_generic_rows(path="generic_seeds.json"):
	seeds = load_generic_seeds(path)
	rows = []
	for seed_id, seed in seeds.items():
		sampled_vars = {name: sample_value(spec) for name, spec in seed["variables"].items()}
		computed_vars = evaluate_formula(seed["formula"], sampled_vars)
		numerics_text = build_numerics_text(sampled_vars, computed_vars, seed["formula"])
		rows.append({
			"id": int(seed_id),
			"persona": seed["persona"],
			"environment": seed["environment"],
			"seed_type": seed["seed_type"],
			"numerics": numerics_text
		})
	return rows