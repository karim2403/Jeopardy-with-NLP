"""Portable, dependency-free HTML, Markdown, and JSON diagnostics reports."""
import html
import json
from pathlib import Path


def tables(report):
    candidates = report["candidates"]
    result = []
    headers = ["Vocab", "Clue mean", "Answer mean", "QA input p95", "Round-trip failures", "Unknown tokens", "Artifact KiB", "Encode strings/s"]
    rows = []
    for c in candidates:
        v = c["validation"]
        rows.append([c["actual_vocab_size"], f'{v["fields"]["clue"]["mean"]:.2f}',
                     f'{v["fields"]["answer"]["mean"]:.2f}', v["fields"]["qa_input"]["p95"],
                     v["roundtrip_failures"], v["unknown_tokens"], f'{c["artifact_bytes"]/1024:.1f}',
                     f'{c["benchmark"]["strings_per_second"]:,.0f}'])
    result.append(("Overview", headers, rows))
    headers = ["Vocab", "Field", "Mean", "p50", "p90", "p95", "p99", "Max", "Tokens/word"]
    rows = []
    for c in candidates:
        for field, s in c["validation"]["fields"].items():
            ratio = s["tokens_per_whitespace_word"]
            rows.append([c["actual_vocab_size"], field, f'{s["mean"]:.2f}',
                         *[s[k] for k in ("p50", "p90", "p95", "p99", "max")],
                         f"{ratio:.2f}" if ratio is not None else "N/A"])
    result.append(("Validation sequence lengths", headers, rows))
    headers = ["Vocab", "Field", ">16", ">32", ">64", ">96", ">128"]
    rows = [[c["actual_vocab_size"], field,
             *[f'{s["over_limit_percent"][str(n)]:.3f}%' for n in (16, 32, 64, 96, 128)]]
            for c in candidates for field, s in c["validation"]["fields"].items() if field in ("qa_input", "answer")]
    result.append(("Sequences exceeding possible length limits", headers, rows))
    headers = ["Vocab", "Training seconds", "128D embedding parameters", "Embedding MiB (FP32)", "Unused tokens", "Tokens used 1–9 times"]
    rows = [[c["actual_vocab_size"], f'{c["training_seconds"]:.2f}', f'{c["embedding_parameters_128d"]:,}',
             f'{c["embedding_parameters_128d"] * 4 / 2**20:.2f}', c["training_usage"]["unused_tokens"],
             c["training_usage"]["used_fewer_than_10_times"]] for c in candidates]
    result.append(("Vocabulary cost and training usage", headers, rows))
    return result


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        f"Trained on {report['train']['rows']:,} training records; evaluated on {report['validation']['rows']:,} validation records. Test data was not read.",
        "Categories, clues, and answers are training text. Raw-field metrics exclude special tokens; QA input is 'Category: {category}\\nClue: {clue}' with BOS and EOS. Answer metrics exclude BOS/EOS: add the tokens your decoder requires when choosing its length limit.",
        "Percentiles use nearest rank. Tokens/word is total tokens divided by total whitespace-separated words; QA input includes labels and BOS/EOS. No truncation or padding was applied to diagnostics.",
        "Speed is median of three warmed-up, single-string encode loops over the same seeded sample of at most 2,000 validation clues. It excludes file I/O and decoding. Timings depend on hardware and background load; small differences are not reliable rankings.",
        "Training usage counts exclude the four special tokens. The full byte alphabet intentionally includes some unused tokens. A 128D embedding table is illustrative, not a trained model; an untied output projection adds approximately another vocab × hidden-dimension parameters, plus bias.",
        "Coverage and compression do not measure semantic quality or answer accuracy. Choose the final vocabulary using downstream validation results, not sequence length alone. Comparisons do not replace the existing default tokenizer artifact.",
        "Token pieces below are the stored byte-level representation: Ġ usually represents a space and Ċ a newline. Other unusual glyphs may represent UTF-8 bytes. Decode the entire ID sequence to recover text; individual byte pieces need not be valid Unicode characters.",
    ]
    esc = lambda value: html.escape(str(value))
    body = ["<h1>Jeopardy tokenizer comparison</h1>", "<p class='subtitle'>BPE vocabulary size • coverage • sequence length • cost</p>"]
    body.append("<section><h2>How to read this report</h2>" + "".join(f"<p>{esc(n)}</p>" for n in notes) + "</section>")
    md = ["# Jeopardy tokenizer comparison", *notes]
    maximum = max(c["validation"]["fields"]["clue"]["mean"] for c in report["candidates"])
    body.append("<section><h2>Mean tokens per clue</h2><p>Lower means shorter sequences, not necessarily better answers.</p>")
    for c in report["candidates"]:
        mean = c["validation"]["fields"]["clue"]["mean"]
        body.append(f"<div class='chartrow'><b>{c['actual_vocab_size']:,}</b><div class='track'><div class='bar' style='width:{100*mean/maximum:.2f}%'>{mean:.2f}</div></div></div>")
    body.append("</section>")
    for title, headers, rows in tables(report):
        body.append(f"<section><h2>{esc(title)}</h2><div class='scroll'><table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>")
        body.extend("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
        body.append("</tbody></table></div></section>")
        md.extend([f"## {title}", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        md.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        md.append("")
    body.append("<section><h2>Token inspector</h2><p>Expand each example to compare the exact token pieces and IDs.</p>")
    for i, example in enumerate(report["candidates"][0]["examples"]):
        body.append(f"<details><summary>{esc(example['label'])}</summary><pre>{esc(example['text'])}</pre>")
        for c in report["candidates"]:
            e = c["examples"][i]
            body.append(f"<h3>{c['actual_vocab_size']:,} vocabulary · {len(e['ids'])} tokens</h3><div class='pieces'>" + "".join(f"<span class='piece' title='ID {token_id}'>{esc(piece)}</span>" for piece, token_id in zip(e["pieces"], e["ids"])) + "</div>")
            body.append(f"<p>IDs: <code>{esc(e['ids'])}</code></p><p>Decoded: {esc(e['decoded'])}</p>")
        body.append("</details>")
    body.append("</section><section><h2>Longest validation sequences</h2>")
    for c in report["candidates"]:
        body.append(f"<details><summary>{c['actual_vocab_size']:,} vocabulary</summary>")
        for field, example in c["validation"]["longest"].items():
            body.append(f"<h3>{esc(field)}: {example['tokens']} tokens · {esc(example['id'])}</h3><pre>{esc(example['text'])}</pre>")
        body.append("</details>")
    body.append("</section><section><h2>Reproducibility</h2><pre>" + esc(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2)) + "</pre></section>")
    css = """body{font:16px/1.55 system-ui,sans-serif;background:#f4f6fa;color:#182a40;margin:0;padding:36px;max-width:1200px;margin:auto}h1{font-size:34px;margin-bottom:0}h2{font-size:23px}h3{font-size:17px}.subtitle{color:#526782}section{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:24px;margin:24px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #e4eaf2;white-space:nowrap}th{background:#edf3fb}pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}pre{background:#f4f6fa;padding:16px;border-radius:8px}.chartrow{display:flex;gap:20px;align-items:center;margin:12px 0}.chartrow b{width:60px}.track{flex:1}.bar{box-sizing:border-box;background:#2864af;color:white;padding:7px 12px;border-radius:5px;min-width:60px}.pieces{display:flex;flex-wrap:wrap;gap:5px}.piece{background:#e5eefb;border:1px solid #bdd1ec;padding:4px 7px;border-radius:5px;font-family:monospace}details{border-top:1px solid #dbe2ec;padding:14px 0}summary{cursor:pointer;font-weight:600}@media(max-width:650px){body{padding:12px}section{padding:14px}}"""
    document = "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Jeopardy tokenizer comparison</title><style>" + css + "</style></head><body>" + "".join(body) + "</body></html>"
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    (output_dir / "report.md").write_text("\n\n".join(md[:1] + notes) + "\n\n" + "\n".join(md[1+len(notes):]) + "\n", encoding="utf-8")
