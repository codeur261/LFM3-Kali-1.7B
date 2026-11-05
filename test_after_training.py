"""Comparaison qualitative sur des requêtes de base ; aucune commande n'est exécutée."""
import json
import time

PROBES = [
    "Affiche la version de Nmap.",
    "Avec Nmap, fais un scan TCP connect des ports 22,80,443 de 192.0.2.30 et affiche seulement les ports ouverts.",
    "Affiche les en-têtes HTTP de http://service.test/ avec curl.",
    "Avec tshark, lis capture.pcap et affiche les paquets DNS.",
    "Explique en français la différence entre un port fermé et un port filtré dans Nmap.",
]
RESULTS = []
model.eval()


@torch.no_grad()
def answer(question):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        tools=SHELL_TOOL, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        ids = model.generate(**inputs, max_new_tokens=160, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


for question in PROBES:
    with model.disable_adapter():
        original = answer(question)
    specialized = answer(question)
    RESULTS.append({"question": question, "base": original, "fine_tuned": specialized})
    print("\nDEMANDE :", question)
    print("BASE :", original)
    print("FINE-TUNÉ :", specialized)

(SAVE_DIR / "qualitative_tests.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nComparaison enregistrée :", SAVE_DIR / "qualitative_tests.json")
