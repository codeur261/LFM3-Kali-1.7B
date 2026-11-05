"""Assemble un corpus traçable. Les commandes sont des données, jamais exécutées."""

import collections
import hashlib
import json
import os
import random
import re
import shlex
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 42
MODEL_ID = "OpenIntelligenceNet/LFM3-1.7B-Uncensored"
MODEL_REVISION = "8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204"
ROOT = Path(os.environ.get("LFM3_WORK_DIR", "/content/lfm3_cyber_ft"))
RAW = ROOT / "raw"
DATA = ROOT / "data"
SYSTEM = (
    "You are an internal cybersecurity assistant. Select tools and arguments from the request "
    "and available evidence. Use the supplied tool schemas. Never invent tool results or claim "
    "to have executed a command without a tool response. Reply in the user's language."
)
SHELL_TOOL = [{"type": "function", "function": {
    "name": "run_shell", "description": "Execute a Kali/Linux command in the working environment.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The exact shell command to execute."}
    }, "required": ["command"], "additionalProperties": False}
}}]

SOURCES = {
    "kali_train.jsonl": {
        "url": "https://raw.githubusercontent.com/RISys-Lab/KaliBench/main/KaliBench_data/kalibench_verified_train_3504.jsonl",
        "sha256": "93dc4ddff6d9df73bed4b9e92716470e7a73df513f5ca2f1b201c85f2329f909", "license": "CC-BY-4.0",
    },
    "kali_test.jsonl": {
        "url": "https://raw.githubusercontent.com/RISys-Lab/KaliBench/main/KaliBench_data/kalibench_verified_test_5000.jsonl",
        "sha256": "783193aaa98762ca384f028b7a7c309994c879ebf0fc1b781dcb7169fff89afc", "license": "CC-BY-4.0",
    },
    "kali_usage.jsonl": {
        "url": "https://raw.githubusercontent.com/RISys-Lab/KaliBench/main/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl",
        "sha256": "4515fc581d624652297344f036cd7f429e8002356956b7b528220097a60254cd", "license": "See upstream Kali tool documentation",
    },
    "agent_train.jsonl": {
        "url": "https://huggingface.co/datasets/hotdogs/cyber-sft-agent-qwen38/resolve/3f3d5a112b8739dc76e77ff771f3df526ef246a4/train.jsonl",
        "sha256": "52f3b1ebd0b8e17aa5c74956b8eb9937b93b436b1daa27e58ab9aead94a649d7", "license": "Apache-2.0",
    },
    "agent_valid.jsonl": {
        "url": "https://huggingface.co/datasets/hotdogs/cyber-sft-agent-qwen38/resolve/3f3d5a112b8739dc76e77ff771f3df526ef246a4/valid.jsonl",
        "sha256": "e7dfdba23fd89552fe73d6cec8e4b7ae29fe9d0f26889a55dd732f6e34e01626", "license": "Apache-2.0",
    },
    "agent_test.jsonl": {
        "url": "https://huggingface.co/datasets/hotdogs/cyber-sft-agent-qwen38/resolve/3f3d5a112b8739dc76e77ff771f3df526ef246a4/test.jsonl",
        "sha256": "76bf523273834a76b12e3fe8a7e90abb97a1f7ca5a86e24dd20758ee87fecc57", "license": "Apache-2.0",
    },
    "hackmentor.json": {
        "url": "https://raw.githubusercontent.com/tmylla/HackMentor/main/data/seed_generation-14k.json",
        "sha256": "8f819f6b7336db5cf4404e2b3f54af36daf961c28bdf253da6c4ef09acc5520b", "license": "Apache-2.0 (upstream repository)",
    },
}


def fetch(name):
    meta = SOURCES[name]
    path = RAW / name
    if not path.exists():
        with urllib.request.urlopen(meta["url"], timeout=120) as response:
            payload = response.read()
        path.write_bytes(payload)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != meta["sha256"]:
        raise ValueError(f"Source modifiée : {name}, attendu {meta['sha256']}, obtenu {actual}")
    print(f"Source vérifiée : {name} ({len(payload)/1e6:.1f} Mo)", flush=True)
    return path


def rows(name):
    text = (RAW / name).read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else [json.loads(x) for x in text.splitlines() if x.strip()]


def shell_example(question, command, source, key):
    return {
        "source": source, "source_id": str(key), "tools": SHELL_TOOL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "run_shell", "arguments": {"command": command}}}
            ]},
        ],
    }


def normalize_query(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def query_key(row):
    return normalize_query(next(m["content"] for m in row["messages"] if m["role"] == "user"))


def make_kali(name, rejected):
    result = []
    for row in rows(name):
        command = row.get("ground_truth_command", "").strip()
        query = row.get("query", "").strip()
        verdict = (row.get("model_verification", {}).get("model_output", {}) or {}).get("verdict")
        if not command or not query or verdict != "ACCURATE":
            rejected["kali_incomplete_or_unverified"] += 1
            continue
        try:
            shlex.split(command)
        except ValueError:
            rejected["kali_unbalanced_shell_quotes"] += 1
            continue
        result.append(shell_example(query, command, "KaliBench", row["custom_id"]))
    return result


CORE_TOOLS = {"nmap", "masscan", "curl", "whatweb", "nikto", "ffuf", "gobuster", "dirb", "wpscan", "searchsploit", "john", "hashcat", "hydra", "sqlmap", "enum4linux"}


def first_tool_calls(name, rejected):
    """Apprend le premier appel uniquement : aucune sortie synthétique contradictoire."""
    result = []
    for row in rows(name):
        if row.get("kind") != "tool":
            continue
        messages = row.get("messages", [])
        first = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), None)
        if first is None:
            continue
        message = messages[first]
        tool_calls = message.get("tool_calls", [])
        if len(tool_calls) != 1:
            continue
        call = tool_calls[0].get("function", tool_calls[0])
        name_call, args = call.get("name"), call.get("arguments")
        definitions = [x.get("function", x) for x in row.get("tools", [])]
        definition = next((x for x in definitions if x.get("name") == name_call), None)
        if name_call not in CORE_TOOLS or not isinstance(args, dict) or definition is None:
            continue
        schema = definition.get("parameters", {})
        props = schema.get("properties", {})
        if set(args) - set(props) or any(x not in args for x in schema.get("required", [])):
            rejected["agent_invalid_arguments"] += 1
            continue
        if any(props[k].get("type") == "string" and not isinstance(v, str) for k, v in args.items()):
            rejected["agent_wrong_argument_type"] += 1
            continue
        user_messages = [{"role": "user", "content": str(x.get("content", ""))} for x in messages[:first] if x.get("role") == "user"]
        if not user_messages:
            continue
        result.append({
            "source": "Cyber-SFT-Agent:first_call_only", "source_id": row["id"],
            "tools": row.get("tools", []),
            "messages": [{"role": "system", "content": SYSTEM}] + user_messages + [
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name_call, "arguments": args}}]}
            ],
        })
    return result


def hackmentor_qa(rejected):
    result = []
    for i, row in enumerate(rows("hackmentor.json")):
        instruction = str(row.get("instruction", "")).strip()
        question = str(row.get("input", "")).strip() or instruction
        answer = str(row.get("output", "")).strip()
        # Questions explicatives seulement ; évite les faux comptes rendus d'audits exécutés.
        if not re.match(r"(?:what\b|explain\b|describe\b|compare\b|define\b|how (?:does|do|can|is|are)\b)", question, re.I):
            continue
        if not 80 <= len(answer) <= 2200 or re.search(r"after conducting|i(?:'ve| have) (?:found|identified|conducted|scanned)|the scan (?:revealed|identified)", answer, re.I):
            rejected["hackmentor_unsupported_audit_claim"] += 1
            continue
        result.append({"source": "HackMentor:explanations", "source_id": str(i), "tools": [], "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}
        ]})
    random.Random(SEED).shuffle(result)
    return result[:1000]


def french_examples():
    result = []
    targets = ["192.0.2.10", "192.0.2.20", "198.51.100.10", "198.51.100.20", "203.0.113.10", "203.0.113.20"]
    ports = ["22,80,443", "80,443", "22,8080", "21,22,25", "53,80,443"]
    for ip in targets:
        for port_list in ports:
            pairs = [
                (f"Avec Nmap, teste en connexion TCP les ports {port_list} de {ip} et affiche les ports ouverts.", f"nmap -sT --open -p {port_list} {ip}"),
                (f"Identifie avec Nmap les versions des services sur les ports {port_list} de {ip}.", f"nmap -sV -p {port_list} {ip}"),
                (f"Affiche uniquement les en-têtes HTTP du serveur http://{ip}/ avec curl.", f"curl -I http://{ip}/"),
                (f"Identifie les technologies du site http://{ip}/ avec WhatWeb.", f"whatweb http://{ip}/"),
                (f"Capture 20 paquets pour l'hôte {ip} sur eth0 avec tcpdump, sans résolution de noms.", f"tcpdump -i eth0 -nn -c 20 host {ip}"),
                (f"Avec tshark, lis capture.pcap et filtre l'adresse IP {ip}.", f"tshark -r capture.pcap -Y 'ip.addr == {ip}'"),
                (f"Liste les suites cryptographiques du service TLS de {ip} sur le port 443 avec Nmap.", f"nmap -sV -p 443 --script ssl-enum-ciphers {ip}"),
            ]
            for question, command in pairs:
                result.append(shell_example(question, command, "FR:curated_tool_templates", len(result)))
    for domain in ["service.test", "audit.test", "interne.test", "laboratoire.test"]:
        for kind in ["A", "AAAA", "MX", "TXT", "NS"]:
            result.append(shell_example(f"Affiche les enregistrements DNS {kind} de {domain} avec dig, en réponse courte.", f"dig +short {domain} {kind}", "FR:curated_tool_templates", len(result)))
    return result


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as stream:
        for row in items:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare():
    RAW.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch, SOURCES))
    rejected = collections.Counter()
    kali_train = make_kali("kali_train.jsonl", rejected)
    kali_test = make_kali("kali_test.jsonl", rejected)
    candidates = kali_train + first_tool_calls("agent_train.jsonl", rejected) + hackmentor_qa(rejected) + french_examples()
    protected = kali_test + first_tool_calls("agent_valid.jsonl", rejected) + first_tool_calls("agent_test.jsonl", rejected)
    protected_keys = {query_key(row) for row in protected}
    dedup, seen = [], set()
    for row in candidates:
        key = query_key(row)
        if key in protected_keys or key in seen:
            rejected["duplicate_or_eval_query"] += 1
            continue
        seen.add(key)
        dedup.append(row)
    train, validation = [], []
    by_source = collections.defaultdict(list)
    for row in dedup:
        by_source[row["source"]].append(row)
    rng = random.Random(SEED)
    for source, group in sorted(by_source.items()):
        rng.shuffle(group)
        count = max(1, round(len(group) * 0.05))
        validation.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "validation.jsonl", validation)
    write_jsonl(DATA / "heldout.jsonl", protected)
    # Les pages d'aide sont conservées pour un RAG ultérieur, pas utilisées comme réponses d'exécution.
    write_jsonl(DATA / "tool_documentation.jsonl", rows("kali_usage.jsonl"))
    assert not ({query_key(x) for x in train} & {query_key(x) for x in validation + protected})
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "seed": SEED, "train": len(train), "validation": len(validation), "heldout": len(protected),
        "training_sources": dict(collections.Counter(x["source"] for x in train)),
        "rejected": dict(rejected), "sources": SOURCES,
        "notes": ["No commands are executed by data preparation or training.", "Original evaluation splits never enter training.", "Synthetic agent outputs and reasoning traces are excluded; only schema-validated first calls are retained.", "No refusal or safety-alignment corpus is added; no zero-refusal guarantee is made.", "French examples use reserved documentation addresses and are generated from deterministic templates."],
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
