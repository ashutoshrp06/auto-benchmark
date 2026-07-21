import json

def load_generic_seeds(path="generic_seeds.json"):
	with open(path, "r") as f:
		return json.load(f)

def get_sampled_generic_rows(path="generic_seeds.json"):
	seeds = load_generic_seeds(path)
	rows = []
	for seed_id, seed in seeds.items():
		rows.append({
			"id": int(seed_id),
			"persona": seed["persona"],
			"environment": seed["environment"],
			"seed_type": seed["seed_type"],
			"numerics": ""
		})
	return rows