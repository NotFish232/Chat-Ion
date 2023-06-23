import json

x = dict(json.load(open("data/openwebtext2/info.json")))

print(f"Number of sentences {x['num_sentences']}")
print(f"Number of passages {x['num_passages']}")
print(f"Vocabulary size {len(x['vocab'])}")

out = [x for x in x['vocab'] if x.isascii()]
print(len(out))

x['vocab'] = out

json.dump(x, open("out.json", "w+"))