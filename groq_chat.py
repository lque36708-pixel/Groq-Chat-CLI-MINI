#!/usr/bin/env python3
"""
Groq Chat CLI — focused on chat, summarization, and file/OCR attachment.
"""

import os
import json
import time
import tempfile
import re
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.theme import Theme

# ── UI Theme ──────────────────────────────────────────────────────────────────
system_theme = Theme({
    "info":     "bright_blue",
    "warning":  "yellow",
    "error":    "bold red",
    "success":  "bold green",
    "user_box": "bold yellow",
    "ai_box":   "bold magenta",
    "label":    "bold cyan",
    "header":   "bold white",
    "dim":      "grey50",
    "search_hl":"bold yellow on dark_red",
})

console = Console(theme=system_theme)

# ── Config paths ──────────────────────────────────────────────────────────────
CONFIG_DIR   = Path.home() / ".groq_chat"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE  = str(CONFIG_DIR / "groq_config.json")
HISTORY_FILE = str(CONFIG_DIR / "input_history")
MAX_HISTORY  = 40

# ── Model list ────────────────────────────────────────────────────────────────
GROQ_MODELS = [
    ("# --- I. DEEP REASONING ---", ""),
    ("qwen/qwen3-32b",                          "Qwen 3 32B – Academic reasoning, technical logic"),
    ("openai/gpt-oss-120b",                     "GPT-OSS 120B – Complex data analysis"),
    ("openai/gpt-oss-20b",                      "GPT-OSS 20B – High-performance reasoning"),
    ("# --- II. LLAMA SERIES ---", ""),
    ("meta-llama/llama-4-scout-17b-16e-instruct","Llama 4 Scout 17B – Instruction following"),
    ("llama-3.3-70b-versatile",                 "Llama 3.3 70B – Versatile, stable (default)"),
    ("llama-3.1-8b-instant",                    "Llama 3.1 8B – Ultra-fast responses"),
    ("# --- III. SPECIALIZED ---", ""),
    ("canopylabs/orpheus-v1-english",            "Orpheus v1 – Text editing"),
    ("allam-2-7b",                              "Allam 2 7B – Lightweight text processing"),
]

# ── System prompt library ─────────────────────────────────────────────────────
PROMPT_LIBRARY = {
    "1": ("General Assistant",       "You are a helpful, concise AI assistant. Answer in the user's language."),
    "2": ("Developer",               "You are an expert software engineer. Provide clean, well-commented code with explanations. Prefer Python unless specified otherwise."),
    "3": ("Creative Writer",         "You are a creative writer. Help craft compelling, vivid prose with strong narrative voice."),
    "4": ("Translator",              "You are a professional translator. Translate accurately while preserving tone and nuance. Do not add commentary unless asked."),
    "5": ("Teacher",                 "You are a patient, clear teacher. Explain concepts step-by-step with simple analogies and examples."),
    "6": ("Data Analyst",            "You are a data analyst. Interpret data, spot patterns, and present insights clearly. Use tables when helpful."),
    "7": ("Vietnam Legal Assistant", "You are a Vietnam legal consultant assistant. Provide accurate information according to current laws, but remind users to consult lawyers for important decisions."),
    "8": ("Summarizer",              "You are an expert at summarizing documents. Produce clear, structured summaries preserving key points, data, and conclusions. Answer in the user's language."),
}

# ── HTML export template ──────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Groq Chat Export – {date}</title>
<style>
  :root {{ --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a; --user:#f0b429; --ai:#a855f7; --text:#e2e8f0; --dim:#64748b; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:2rem; max-width:860px; margin:auto; }}
  h1 {{ color:var(--ai); font-size:1.4rem; margin-bottom:.3rem; }}
  .meta {{ color:var(--dim); font-size:.85rem; margin-bottom:2rem; border-bottom:1px solid var(--border); padding-bottom:1rem; }}
  .msg {{ margin-bottom:1.5rem; border-radius:10px; overflow:hidden; border:1px solid var(--border); }}
  .msg-header {{ padding:.5rem 1rem; font-size:.8rem; font-weight:700; letter-spacing:.05em; }}
  .user .msg-header {{ background:rgba(240,180,41,.15); color:var(--user); }}
  .ai   .msg-header {{ background:rgba(168,85,247,.15); color:var(--ai);   }}
  .msg-body {{ padding:1rem; background:var(--surface); line-height:1.7; white-space:pre-wrap; font-size:.95rem; }}
  code {{ background:#0d1117; padding:.15em .4em; border-radius:4px; font-size:.88em; color:#7dd3fc; }}
  pre  {{ background:#0d1117; padding:1rem; border-radius:8px; overflow-x:auto; margin:.8rem 0; }}
  pre code {{ background:none; padding:0; }}
</style>
</head>
<body>
<h1>💬 Groq Chat Export</h1>
<div class="meta">📅 {date} &nbsp;|&nbsp; 🤖 Model: {model} &nbsp;|&nbsp; 💬 {turns} turns</div>
{messages}
</body></html>"""


# ── Tesseract check ───────────────────────────────────────────────────────────
def check_tesseract() -> bool:
    import shutil
    from rich.panel import Panel
    from rich.box import ROUNDED
    if not shutil.which("tesseract"):
        console.print(Panel(
            "[bold red]WARNING: Tesseract OCR not found![/bold red]\n\n"
            "Scanned PDF and image OCR will be disabled.\n\n"
            "[bold]Install on Linux:[/bold]\n"
            "  sudo apt-get install tesseract-ocr tesseract-ocr-vie\n\n"
            "[bold]Install on Windows:[/bold]\n"
            "  https://github.com/UB-Mannheim/tesseract/wiki",
            title="Configuration Warning",
            border_style="warning",
            box=ROUNDED,
        ))
        return False
    return True


class GroqChat:
    def __init__(self):
        self.api_key       = ""
        self.model         = "llama-3.3-70b-versatile"
        self.system_prompt = "You are a helpful, concise AI assistant. Answer in the user's language."
        self.history       = []
        self.temperature   = 0.7
        self.api_url       = "https://api.groq.com/openai/v1/chat/completions"
        self.tesseract_ok  = check_tesseract()
        self.session_calls = 0
        self.working_dirs  = []   # [{"path": str, "name": str}]
        self.kb            = None

        self.load_config()

    # ── Key bindings ──────────────────────────────────────────────────────────
    def _setup_keybindings(self):
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.filters import completion_is_selected
        self.kb = KeyBindings()

        @self.kb.add('c-j')
        def _(event):
            event.current_buffer.validate_and_handle()

        @self.kb.add('escape', 'enter')
        def _(event):
            pass

        @self.kb.add('enter', filter=completion_is_selected)
        def _(event):
            event.current_buffer.complete_state = None

        @self.kb.add('c-q')
        def _(event):
            from rich.live import Live
            from rich.spinner import Spinner
            console.print()
            with Live(Spinner("dots", text="[bold cyan]Good Bye![/bold cyan]"), refresh_per_second=10):
                time.sleep(0.5)
            event.app.exit()

    # ── Config ────────────────────────────────────────────────────────────────
    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.api_key       = cfg.get("api_key", "")
            self.model         = cfg.get("model", self.model)
            self.system_prompt = cfg.get("system_prompt", self.system_prompt)
            self.temperature   = float(cfg.get("temperature", 0.7))
            # Migration: support old string-only format
            raw_dirs = cfg.get("working_dirs", [])
            self.working_dirs = []
            for item in raw_dirs:
                if isinstance(item, str):
                    self.working_dirs.append({"path": item, "name": Path(item).name})
                else:
                    self.working_dirs.append(item)
        except Exception as e:
            console.print(f"[error]Error reading config: {e}[/error]")

    def save_config(self):
        cfg = {
            "api_key":       self.api_key,
            "model":         self.model,
            "system_prompt": self.system_prompt,
            "temperature":   self.temperature,
            "working_dirs":  self.working_dirs,
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            console.print(f"[error]Error saving config: {e}[/error]")

    # ── Working directories ───────────────────────────────────────────────────
    def cmd_set_working_dir(self, arg: str):
        import shlex
        from rich.table import Table
        from rich.box import ROUNDED
        if not arg:
            if not self.working_dirs:
                console.print("[warning]No working directories saved. Use: /wd <path> <alias>[/warning]")
                return
            t = Table(title="Working Directories", box=ROUNDED)
            t.add_column("Alias", style="label")
            t.add_column("Path",  style="dim")
            for d in self.working_dirs:
                t.add_row(d["name"], d["path"])
            console.print(t)
            return
        try:
            parts = shlex.split(arg)
            if len(parts) < 2:
                console.print("[error]Syntax: /wd <path> <alias>  (alias has no spaces)[/error]")
                return
            path_raw, name_raw = parts[0], parts[1]
            if " " in name_raw:
                console.print(f"[error]Alias cannot contain spaces: '{name_raw}'[/error]")
                return
            new_path = Path(path_raw).expanduser().resolve()
            if not new_path.is_dir():
                console.print(f"[error]Not a directory: {path_raw}[/error]")
                return
            if any(d["name"] == name_raw for d in self.working_dirs):
                console.print(f"[error]Alias '{name_raw}' already exists.[/error]")
                return
            self.working_dirs.append({"path": str(new_path), "name": name_raw})
            self.save_config()
            console.print(f"[success]Added: [bold]{name_raw}[/bold] → {new_path}[/success]")
        except Exception as e:
            console.print(f"[error]{e}[/error]")

    def cmd_remove_working_dir(self, arg: str):
        if not arg:
            console.print("[warning]Syntax: /wd-rm <alias>[/warning]")
            return
        before = len(self.working_dirs)
        self.working_dirs = [d for d in self.working_dirs if d["name"] != arg]
        if len(self.working_dirs) < before:
            self.save_config()
            console.print(f"[success]Removed: {arg}[/success]")
        else:
            console.print(f"[error]Alias not found: {arg}[/error]")

    def cmd_list_wd(self):
        from rich.table import Table
        from rich.box import ROUNDED
        if not self.working_dirs:
            console.print("[warning]No working directories saved.[/warning]")
            return
        t = Table(title="Working Directories", box=ROUNDED)
        t.add_column("Alias", style="label")
        t.add_column("Path",  style="dim")
        for d in self.working_dirs:
            t.add_row(d["name"], d["path"])
        console.print(t)

    def greedy_search_file(self, dir_name: str, query: str) -> str:
        """
        Smart file finder inside a registered working directory.
        Strategy:
          1. Exact filename match (case-insensitive)
          2. All query tokens present in filename
          3. RapidFuzz WRatio on filename, boosted by recency
        Returns the full path string of the best match.
        """
        from rapidfuzz import fuzz, utils as rfutils
        import time as _time

        target_dir = next((Path(d["path"]) for d in self.working_dirs if d["name"] == dir_name), None)
        if not target_dir:
            raise ValueError(f"Alias '{dir_name}' not found. Register with /wd <path> {dir_name}")

        IGNORE = {".git", "node_modules", "__pycache__", "venv", ".venv", "$RECYCLE.BIN"}
        files: list[Path] = []
        for p in target_dir.rglob("*"):
            if p.is_file() and not any(part in IGNORE for part in p.parts):
                files.append(p)

        if not files:
            raise ValueError(f"No files found in '{dir_name}'.")

        query_norm  = rfutils.default_process(query)
        query_tokens = query_norm.split()
        now = _time.time()

        def score(p: Path) -> float:
            name_norm = rfutils.default_process(p.name)
            # Exact match → perfect score
            if query_norm == name_norm:
                return 200.0
            # All tokens present in name → high score
            token_hit = all(t in name_norm for t in query_tokens)
            base = fuzz.WRatio(query_norm, name_norm)
            # Partial path bonus: query tokens found in parent dirs
            path_norm = rfutils.default_process(str(p.relative_to(target_dir)))
            path_bonus = 10 if any(t in path_norm for t in query_tokens) else 0
            token_bonus = 20 if token_hit else 0
            # Recency bonus: files modified within last 7 days get up to +15
            try:
                age_days = (now - p.stat().st_mtime) / 86400
                recency_bonus = max(0, 15 - age_days * 2)
            except Exception:
                recency_bonus = 0
            return base + token_bonus + path_bonus + recency_bonus

        best = max(files, key=score)
        best_score = score(best)

        if best_score < 30:
            raise ValueError(f"No confident match for '{query}' (best score: {best_score:.1f}).")

        rel = best.relative_to(target_dir)
        console.print(f"[dim]Auto-matched: [bold]{rel}[/bold] (score {best_score:.1f})[/dim]")
        return str(best)

    # ── File / URL extraction ─────────────────────────────────────────────────
    def extract_file_content(self, path: str) -> tuple[str, str]:
        """Return (content, label). Supports URL, txt, md, py, docx, xlsx, pdf, images."""
        import requests as req

        path = path.strip().strip("'\"")

        # URL
        if path.startswith("http://") or path.startswith("https://"):
            fetch_url = f"https://markdown.download/{path}"
            resp = req.get(fetch_url, timeout=30, headers={"User-Agent": "groq-chat/4.0"})
            if resp.status_code != 200:
                raise ValueError(f"Cannot load page (HTTP {resp.status_code}): {path}")
            content = resp.text.strip()
            if not content:
                raise ValueError("Page returned empty content.")
            console.print("[success]Extracted text from webpage.[/success]")
            return content, path

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        # .docx
        if ext == ".docx":
            import docx
            doc = docx.Document(path)
            console.print("[success]Extracted text from .docx.[/success]")
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()), os.path.basename(path)

        # .xlsx
        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, data_only=True)
            lines = []
            for ws in wb.worksheets:
                lines.append(f"--- Sheet: {ws.title} ---")
                for row in ws.iter_rows(values_only=True):
                    row_str = " | ".join(str(c) if c is not None else "" for c in row)
                    if row_str.strip(" |"):
                        lines.append(row_str)
            console.print("[success]Extracted data from .xlsx.[/success]")
            return "\n".join(lines), os.path.basename(path)

        # .pdf
        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(path)
                digital_text = "".join(doc.load_page(i).get_text() for i in range(len(doc)))

                if digital_text.strip():
                    doc.close()
                    console.print("[success]Extracted text from PDF.[/success]")
                    return digital_text, os.path.basename(path)

                # Scanned PDF → OCR
                if not self.tesseract_ok:
                    doc.close()
                    raise RuntimeError("Tesseract not available for scanned PDF.")

                from PIL import Image
                import pytesseract
                console.print("[info]Scanned PDF detected — running OCR...[/info]")
                ocr_text = ""
                for i in range(len(doc)):
                    pix = doc.load_page(i).get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    console.print(f"[info]  Page {i+1}/{len(doc)}...[/info]")
                    ocr_text += pytesseract.image_to_string(img, lang="vie+eng") + "\n"
                doc.close()

                if not ocr_text.strip():
                    raise ValueError("OCR returned no text.")
                console.print("[success]OCR complete.[/success]")
                return ocr_text, os.path.basename(path)

            except Exception as e:
                raise RuntimeError(f"PDF processing error: {e}")

        # Images
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}:
            if not self.tesseract_ok:
                raise RuntimeError("Tesseract not available for image OCR.")
            from PIL import Image
            import pytesseract
            console.print("[info]Running OCR on image...[/info]")
            text = pytesseract.image_to_string(Image.open(path), lang="vie+eng")
            if not text.strip():
                raise ValueError("OCR returned no text from image.")
            console.print("[success]OCR complete.[/success]")
            return text, os.path.basename(path)

        # Plain text fallback
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            console.print("[success]Read text file.[/success]")
            return content, os.path.basename(path)
        except Exception as e:
            raise RuntimeError(f"Cannot read file: {e}")

    # ── API call ──────────────────────────────────────────────────────────────
    def call_api(self, user_input: str):
        import requests as req
        from rich.panel import Panel
        from rich.box import ROUNDED
        from rich.markdown import Markdown

        if not self.api_key:
            console.print("[error]No API key. Use /set-key <key>[/error]")
            return

        console.print(Panel(user_input, title="[user_box]YOU[/user_box]", border_style="user_box", box=ROUNDED))

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.history[-MAX_HISTORY:]
            + [{"role": "user", "content": user_input}]
        )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "stream":      True,
        }

        full_content = ""
        self.session_calls += 1

        try:
            with console.status("[info]Thinking...[/info]", spinner="dots"):
                response = req.post(self.api_url, headers=headers, json=payload, stream=True, timeout=90)

            if response.status_code >= 300:
                try:
                    err_msg = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                if response.status_code == 429:
                    console.print(Panel(
                        f"[warning]Rate limit reached.[/warning]\n[dim]{err_msg}[/dim]",
                        title="Rate Limit", border_style="warning", box=ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        f"[error]HTTP {response.status_code}[/error]\n{err_msg}",
                        title="API Error", border_style="error", box=ROUNDED,
                    ))
                return

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                raw = decoded[6:]
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        full_content += delta["content"]
                except Exception:
                    continue

        except Exception as e:
            console.print(f"[error]Request failed: {e}[/error]")
            return

        if full_content.strip():
            # Strip <think> tags — show reasoning separately if present
            think_match = re.search(r"<think>(.*?)</think>", full_content, re.DOTALL)
            answer = re.sub(r"<think>.*?</think>", "", full_content, flags=re.DOTALL).strip()

            console.print(f"[ai_box]ASSISTANT:[/ai_box]")
            if answer:
                console.print(Markdown(answer))

            self.history.append({"role": "user",      "content": user_input})
            self.history.append({"role": "assistant",  "content": full_content})

        print()

    # ── Commands ──────────────────────────────────────────────────────────────
    def cmd_view(self):
        last_ai = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), None)
        if not last_ai:
            console.print("[warning]No AI response to view.[/warning]")
            return

        processed = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", last_ai, flags=re.DOTALL)
        processed = re.sub(r"\\\((.*?)\\\)", r"$\1$", processed)
        processed = re.sub(
            r"<think>(.*?)</think>",
            lambda m: (
                f'<details style="margin:1rem 0;background:#252830;padding:1rem;border-radius:8px;">'
                f'<summary style="cursor:pointer;font-weight:bold;color:#facc15;">View reasoning</summary>'
                f'<div style="margin-top:.5rem;">{m.group(1).strip()}</div></details>'
            ),
            processed, flags=re.DOTALL,
        )

        json_content = json.dumps(processed)
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
<script>window.MathJax={{tex:{{inlineMath:[["$","$"]],displayMath:[["$$","$$"]],processEscapes:true}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body{{background:#0f1117;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:2rem;line-height:1.6;display:flex;flex-direction:column;min-height:100vh;margin:0;}}
  #logo{{color:#facc15;font-size:1.5rem;font-weight:bold;margin-bottom:2rem;}}
  .rainbow-text{{animation:rainbow 3s linear infinite;}}
  @keyframes rainbow{{0%{{color:#facc15;}}33%{{color:#a855f7;}}66%{{color:#7dd3fc;}}100%{{color:#facc15;}}}}
  #app-container{{max-width:900px;width:100%;margin:0 auto;flex:1;background:#1a1c23;padding:3rem;border-radius:16px;box-shadow:0 10px 25px -5px rgba(0,0,0,.3);}}
  pre{{background:#2d2d2d;padding:1.5rem;border-radius:8px;overflow-x:auto;position:relative;}}
  code{{font-family:Consolas,Monaco,'Andale Mono','Ubuntu Mono',monospace;}}
  .copy-btn{{cursor:pointer;background:#444;color:#fff;border:none;padding:4px 8px;border-radius:4px;position:absolute;top:5px;right:5px;font-size:12px;}}
  h1,h2,h3{{color:#facc15;}}
  table{{width:100%;border-collapse:collapse;margin:1rem 0;}}
  th,td{{border:1px solid #444;padding:.75rem;}}
</style></head><body>
<div id="logo"><span class="rainbow-text">></span>GROQ CHAT CLI MINI</div>
<div id="app-container"><div id="content"></div></div>
<script>
  document.getElementById("content").innerHTML=marked.parse({json_content});
  document.querySelectorAll("pre").forEach(pre=>{{
    if(!pre.querySelector("code"))return;
    let btn=document.createElement("button");btn.className="copy-btn";btn.innerText="Copy";
    btn.onclick=()=>{{navigator.clipboard.writeText(pre.querySelector("code").innerText);btn.innerText="Copied!";setTimeout(()=>btn.innerText="Copy",2000);}};
    pre.prepend(btn);
  }});
  MathJax.typesetPromise().then(()=>Prism.highlightAll());
</script></body></html>"""

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.write(html.encode("utf-8"))
        tmp.close()
        os.system(f"xdg-open {tmp.name}" if os.name != "nt" else f"start {tmp.name}")

    def cmd_save(self):
        if not self.history:
            console.print("[warning]History is empty.[/warning]")
            return
        fname = f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"# Groq Chat Export\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  \n")
            f.write(f"**Model:** {self.model}  \n\n---\n\n")
            for m in self.history:
                role = "### 👤 User" if m["role"] == "user" else "### 🤖 Assistant"
                content = m.get("content") or ""
                f.write(f"{role}\n\n{content}\n\n---\n\n")
        console.print(f"[success]Saved: {fname}[/success]")

    def cmd_export_html(self):
        if not self.history:
            console.print("[warning]History is empty.[/warning]")
            return
        import html as htmllib
        msgs_html = ""
        for m in self.history:
            if m["role"] not in ("user", "assistant"):
                continue
            role_cls  = "user" if m["role"] == "user" else "ai"
            role_name = "👤 User" if m["role"] == "user" else "🤖 Assistant"
            content   = htmllib.escape(m.get("content") or "")
            content   = re.sub(r"```(\w*)\n(.*?)```", r"<pre><code>\2</code></pre>", content, flags=re.DOTALL)
            content   = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            content   = content.replace("\n", "<br>")
            msgs_html += f'<div class="msg {role_cls}"><div class="msg-header">{role_name}</div><div class="msg-body">{content}</div></div>\n'

        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        turns = sum(1 for m in self.history if m["role"] == "user")
        html_out = HTML_TEMPLATE.format(date=now, model=self.model, turns=turns, messages=msgs_html)
        fname = f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html_out)
        console.print(f"[success]Exported HTML: {fname}[/success]")

    def cmd_copy(self):
        last_ai = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), None)
        if not last_ai:
            console.print("[warning]No response to copy.[/warning]")
            return
        try:
            import pyperclip
            pyperclip.copy(last_ai)
            console.print(f"[success]Copied {len(last_ai)} chars to clipboard.[/success]")
        except ImportError:
            console.print("[warning]pip install pyperclip for clipboard support.[/warning]")
            console.print(last_ai)

    def cmd_history(self):
        from rich.table import Table
        from rich.box import SIMPLE
        if not self.history:
            console.print("[warning]History is empty.[/warning]")
            return
        t = Table(box=SIMPLE, show_header=True, header_style="bold cyan", expand=True)
        t.add_column("#",    style="dim",   width=4)
        t.add_column("Role", style="label", width=12)
        t.add_column("Content")
        for i, m in enumerate(self.history, 1):
            if m["role"] == "user":
                label = "[user_box]User[/user_box]"
            elif m["role"] == "assistant":
                label = "[ai_box]Assistant[/ai_box]"
            else:
                continue
            preview = (m.get("content") or "")[:200].replace("\n", " ")
            t.add_row(str(i), label, preview)
        console.print(t)
        console.print(f"[dim]Total: {len(self.history)} messages[/dim]")

    def cmd_search(self, keyword: str):
        from rich.panel import Panel
        from rich.box import ROUNDED
        if not keyword:
            console.print("[warning]Syntax: /search <keyword>[/warning]")
            return
        kw = keyword.lower()
        results = [(i, m) for i, m in enumerate(self.history, 1) if kw in (m.get("content") or "").lower()]
        if not results:
            console.print(f"[warning]'{keyword}' not found in history.[/warning]")
            return
        console.print(f"[info]Found {len(results)} result(s):[/info]")
        for idx, m in results:
            text = (m.get("content") or "")[:400]
            highlighted = text.replace(keyword, f"[search_hl]{keyword}[/search_hl]")
            console.print(Panel(highlighted, title=f"[dim]#{idx}[/dim] {m['role'].capitalize()}", box=ROUNDED, border_style="dim"))

    def cmd_retry(self):
        last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), None)
        if not last_user:
            console.print("[warning]No message to retry.[/warning]")
            return
        if len(self.history) >= 2 and self.history[-1]["role"] == "assistant":
            self.history = self.history[:-2]
        elif self.history and self.history[-1]["role"] == "user":
            self.history = self.history[:-1]
        console.print(f"[info]Retrying: {last_user[:80]}…[/info]")
        self.call_api(last_user)

    def cmd_models(self):
        from rich.table import Table
        from rich.box import ROUNDED
        t = Table(title="Groq Models", box=ROUNDED, show_header=True, header_style="bold cyan")
        t.add_column("#",        style="dim", width=4)
        t.add_column("Model ID", style="label", min_width=35)
        t.add_column("Notes",    min_width=30)
        counter = 1
        for mid, note in GROQ_MODELS:
            if mid.startswith("#"):
                t.add_row("", f"[bold yellow]{mid.replace('# ', '')}[/bold yellow]", "")
                continue
            marker = " ✓" if mid == self.model else ""
            t.add_row(str(counter), mid + f"[green]{marker}[/green]", note)
            counter += 1
        console.print(t)
        console.print("[dim]Enter number to select, or Enter to skip:[/dim] ", end="")
        try:
            choice = input().strip()
            if choice.isdigit():
                models_only = [m for m in GROQ_MODELS if not m[0].startswith("#")]
                idx = int(choice) - 1
                if 0 <= idx < len(models_only):
                    self.model = models_only[idx][0]
                    self.save_config()
                    console.print(f"[success]Model: {self.model}[/success]")
                else:
                    console.print("[warning]Invalid number.[/warning]")
        except (EOFError, KeyboardInterrupt):
            pass

    def cmd_prompt_lib(self):
        from rich.table import Table
        from rich.box import ROUNDED
        t = Table(title="System Prompt Library", box=ROUNDED)
        t.add_column("#",    style="dim",   width=4)
        t.add_column("Name", style="label", min_width=20)
        t.add_column("Preview")
        for num, (name, text) in PROMPT_LIBRARY.items():
            t.add_row(num, name, text[:70] + "…")
        console.print(t)
        console.print("[dim]Enter number to select, or Enter to skip:[/dim] ", end="")
        try:
            choice = input().strip()
            if choice in PROMPT_LIBRARY:
                name, text = PROMPT_LIBRARY[choice]
                self.system_prompt = text
                self.save_config()
                console.print(f"[success]Prompt set: {name}[/success]")
        except (EOFError, KeyboardInterrupt):
            pass

    def cmd_info(self):
        from rich.table import Table
        from rich.box import ROUNDED
        t = Table(title="Current Configuration", box=ROUNDED)
        t.add_column("Parameter", style="label")
        t.add_column("Value")
        t.add_row("Model",        self.model)
        t.add_row("Temperature",  str(self.temperature))
        t.add_row("API Key",      ("✓ Set (" + self.api_key[:8] + "…)") if self.api_key else "[error]Not set[/error]")
        t.add_row("Tesseract",    "[success]Available[/success]" if self.tesseract_ok else "[warning]Not found[/warning]")
        t.add_row("System Prompt",self.system_prompt[:80] + ("…" if len(self.system_prompt) > 80 else ""))
        t.add_row("API calls",    str(self.session_calls))
        t.add_row("Messages",     str(len(self.history)))
        if self.working_dirs:
            t.add_row("Working dirs", ", ".join(d["name"] for d in self.working_dirs))
        console.print(t)

    def show_help(self):
        from rich.table import Table
        from rich.panel import Panel
        from rich.box import ROUNDED, SIMPLE
        t = Table(title="GROQ CHAT CLI MINI — Commands", title_style="header", show_header=True,
                  header_style="bold cyan", box=ROUNDED)
        t.add_column("Command",     style="label", min_width=22)
        t.add_column("Description", min_width=50)

        rows = [
            ("── Conversation ──",    ""),
            ("/retry",                "Re-send the last message"),
            ("/clear",                "Clear chat history"),
            ("/clear-scr",            "Clear terminal screen"),
            ("/history",              "View full chat history"),
            ("/search <keyword>",     "Search keyword in chat history"),
            ("── Files ──",           ""),
            ("/attach <path|url>",           "Attach file or webpage into context"),
            ("/attach --find-in:<alias> <q>","Fuzzy-search file by name in a working dir and attach"),
            ("/attach --latest-in:<alias>",  "Attach the most recently modified file in a working dir"),
            ("/unattach",                    "Remove last attached file from context"),
            ("── Working Directories ──",    ""),
            ("/wd <path> <alias>",           "Register a directory with a short alias"),
            ("/wd",                          "List all registered directories"),
            ("/ls-wd",                       "List all registered working directory paths"),
            ("/wd-rm <alias>",               "Remove a registered directory"),
            ("── Display & Export ──",""),
            ("/view",                 "Open last AI response in browser (MathJax + code highlighting)"),
            ("/save",                 "Save conversation as Markdown file"),
            ("/export html",          "Export conversation as styled HTML file"),
            ("/copy",                 "Copy last AI response to clipboard"),
            ("── Settings ──",        ""),
            ("/models",               "Browse and select a Groq model"),
            ("/model <id>",           "Switch model directly by ID"),
            ("/temp <0.0–2.0>",       "Set temperature (0 = precise, 2 = creative)"),
            ("/system",               "View current system prompt"),
            ("/system <text>",        "Set a custom system prompt"),
            ("/prompt-lib",           "Pick a system prompt from the built-in library"),
            ("/set-key <key>",        "Set or update Groq API key"),
            ("/info",                 "Show current configuration"),
            ("── General ──",         ""),
            ("/help",                 "Show this help menu"),
            ("/bye",                  "Exit"),
        ]
        for cmd, desc in rows:
            if desc == "":
                t.add_row(f"[dim]{cmd}[/dim]", "")
            else:
                t.add_row(cmd, desc)
        console.print(t)
        console.print(Panel(
            f"Model: [green]{self.model}[/green]  |  Temp: [cyan]{self.temperature}[/cyan]  |  "
            f"[dim]Ctrl+J to send | Enter for newline | Ctrl+Q to quit[/dim]",
            border_style="info", box=ROUNDED,
        ))

    def get_completer(self):
        from prompt_toolkit.completion import NestedCompleter
        wd_aliases = {d["name"]: None for d in self.working_dirs}
        cmds = {
            "/attach":     {
                "--find-in:"  + d["name"]: None for d in self.working_dirs
            } | {
                "--latest-in:" + d["name"]: None for d in self.working_dirs
            },
            "/unattach":   None,
            "/wd":         None,
            "/ls-wd":      None,
            "/wd-rm":      wd_aliases,
            "/view":       None,
            "/save":       None,
            "/export":     {"html": None},
            "/copy":       None,
            "/history":    None,
            "/search":     None,
            "/retry":      None,
            "/clear":      None,
            "/clear-scr":  None,
            "/model":      {m[0]: None for m in GROQ_MODELS if not m[0].startswith("#")},
            "/models":     None,
            "/temp":       None,
            "/system":     None,
            "/prompt-lib": None,
            "/set-key":    None,
            "/info":       None,
            "/help":       None,
            "/bye":        None,
        }
        return NestedCompleter.from_nested_dict(cmds)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        from prompt_toolkit import prompt
        from prompt_toolkit.styles import Style as PromptStyle
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from rich.panel import Panel
        from rich.box import ROUNDED

        if self.kb is None:
            self._setup_keybindings()

        # Banner
        art = [
            " ▝▜▄     [bold]Groq Chat CLI Mini[/bold]",
            "   ▝▜▄   [dim](MINI VERSION)[/dim]",
            "  ▗▟▀    [bold]Ready[/bold]",
            f" ▝▀      [bold]Model: {self.model}[/bold]  /models",
        ]
        colors = ["blue", "bright_blue", "cyan", "magenta", "bright_magenta"]
        for i, line in enumerate(art):
            console.print(f"[{colors[i % len(colors)]}]{line}[/{colors[i % len(colors)]}]")
        console.print("\n[dim]Type [bold cyan]/help[/bold cyan] for commands | [bold cyan]Ctrl+J[/bold cyan] to send | [bold cyan]Ctrl+Q[/bold cyan] to quit[/dim]\n")

        prompt_style = PromptStyle.from_dict({"prompt": "bold yellow"})
        file_history = FileHistory(HISTORY_FILE)

        while True:
            try:
                user_input = prompt(
                    "➜ ",
                    key_bindings=self.kb,
                    multiline=True,
                    style=prompt_style,
                    history=file_history,
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=self.get_completer(),
                    complete_while_typing=True,
                    complete_in_thread=True,
                )

                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if not user_input.startswith("/"):
                    self.call_api(user_input)
                    continue

                parts = user_input.split(" ", 1)
                cmd   = parts[0].lower()
                arg   = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/bye":
                    from rich.live import Live
                    from rich.spinner import Spinner
                    with Live(Spinner("dots", text="[bold cyan]Good Bye![/bold cyan]"), refresh_per_second=10):
                        time.sleep(0.5)
                    break

                elif cmd == "/set-key":
                    if not arg:
                        console.print("[warning]Syntax: /set-key <api_key>[/warning]")
                    else:
                        self.api_key = arg
                        self.save_config()
                        console.print("[success]API key saved.[/success]")

                elif cmd == "/model":
                    if not arg:
                        console.print(f"[info]Current model: {self.model}[/info]")
                    else:
                        self.model = arg
                        self.save_config()
                        console.print(f"[success]Model: {self.model}[/success]")

                elif cmd == "/models":
                    self.cmd_models()

                elif cmd == "/temp":
                    try:
                        val = float(arg)
                        if not 0.0 <= val <= 2.0:
                            raise ValueError
                        self.temperature = val
                        self.save_config()
                        console.print(f"[success]Temperature: {val}[/success]")
                    except ValueError:
                        console.print("[warning]Syntax: /temp <0.0 – 2.0>[/warning]")

                elif cmd == "/system":
                    if not arg:
                        console.print(f"[info]System prompt: {self.system_prompt}[/info]")
                    else:
                        self.system_prompt = arg
                        self.save_config()
                        console.print("[success]System prompt updated.[/success]")

                elif cmd == "/prompt-lib":
                    self.cmd_prompt_lib()

                elif cmd == "/attach":
                    from rich.panel import Panel
                    from rich.box import ROUNDED
                    if not arg:
                        console.print("[warning]Syntax: /attach <path|url>[/warning]")
                        console.print("[warning]        /attach --find-in:<alias> <query>[/warning]")
                        console.print("[warning]        /attach --latest-in:<alias>[/warning]")
                    elif arg.startswith("--latest-in:"):
                        alias = arg[len("--latest-in:"):].strip()
                        try:
                            target = next((Path(d["path"]) for d in self.working_dirs if d["name"] == alias), None)
                            if not target:
                                raise ValueError(f"Alias '{alias}' not found.")
                            files = [p for p in target.rglob("*") if p.is_file()]
                            if not files:
                                raise ValueError("Directory is empty.")
                            latest = max(files, key=lambda p: p.stat().st_mtime)
                            with console.status("[info]Reading file...[/info]"):
                                content, label = self.extract_file_content(str(latest))
                            if content:
                                self.history.append({"role": "user", "content": f"[Attached: {label}]\n{content}"})
                                console.print(Panel(
                                    f"📄 Attached latest: [bold]{label}[/bold] ({len(content.encode())/1024:.1f} KB)",
                                    border_style="success", box=ROUNDED,
                                ))
                        except Exception as e:
                            console.print(f"[error]{e}[/error]")
                    elif arg.startswith("--find-in:"):
                        rest = arg[len("--find-in:"):].strip()
                        parts2 = rest.split(" ", 1)
                        if len(parts2) < 2:
                            console.print("[error]Syntax: /attach --find-in:<alias> <query>[/error]")
                        else:
                            alias, query = parts2[0], parts2[1].strip()
                            try:
                                with console.status(f"[info]Searching in '{alias}'...[/info]"):
                                    found_path = self.greedy_search_file(alias, query)
                                    content, label = self.extract_file_content(found_path)
                                if content:
                                    self.history.append({"role": "user", "content": f"[Attached: {label}]\n{content}"})
                                    console.print(Panel(
                                        f"📄 Attached: [bold]{label}[/bold] ({len(content.encode())/1024:.1f} KB)",
                                        border_style="success", box=ROUNDED,
                                    ))
                            except Exception as e:
                                console.print(f"[error]{e}[/error]")
                    else:
                        is_url = arg.startswith("http://") or arg.startswith("https://")
                        status_msg = "[info]Loading webpage...[/info]" if is_url else "[info]Reading file...[/info]"
                        try:
                            with console.status(status_msg):
                                content, label = self.extract_file_content(arg)
                            if content:
                                size_kb = len(content.encode()) / 1024
                                self.history.append({"role": "user", "content": f"[Attached: {label}]\n{content}"})
                                icon = "🌐" if is_url else "📄"
                                console.print(Panel(
                                    f"{icon} Attached: [bold]{label}[/bold] ({size_kb:.1f} KB)",
                                    border_style="success", box=ROUNDED,
                                ))
                        except Exception as e:
                            console.print(f"[error]{e}[/error]")

                elif cmd == "/unattach":
                    for i in range(len(self.history) - 1, -1, -1):
                        if self.history[i]["role"] == "user" and self.history[i]["content"].startswith("[Attached:"):
                            label = self.history[i]["content"].split("\n")[0]
                            self.history.pop(i)
                            console.print(f"[success]Removed: {label}[/success]")
                            break
                    else:
                        console.print("[warning]No attached files in history.[/warning]")

                elif cmd == "/wd":
                    self.cmd_set_working_dir(arg)

                elif cmd == "/ls-wd":
                    self.cmd_list_wd()

                elif cmd == "/wd-rm":
                    self.cmd_remove_working_dir(arg)

                elif cmd == "/view":
                    self.cmd_view()

                elif cmd == "/save":
                    self.cmd_save()

                elif cmd == "/export" and arg.lower() == "html":
                    self.cmd_export_html()

                elif cmd == "/copy":
                    self.cmd_copy()

                elif cmd == "/history":
                    self.cmd_history()

                elif cmd == "/search":
                    self.cmd_search(arg)

                elif cmd == "/retry":
                    self.cmd_retry()

                elif cmd == "/clear":
                    self.history = []
                    console.print("[info]History cleared.[/info]")

                elif cmd == "/clear-scr":
                    os.system("cls" if os.name == "nt" else "clear")
                    for i, line in enumerate(art):
                        console.print(f"[{colors[i % len(colors)]}]{line}[/{colors[i % len(colors)]}]")
                    console.print()

                elif cmd == "/info":
                    self.cmd_info()

                elif cmd == "/help":
                    self.show_help()

                else:
                    console.print(f"[error]Unknown command: {cmd}[/error]  Use /help to see all commands.")

            except (EOFError, SystemExit):
                break
            except KeyboardInterrupt:
                print()
                continue


if __name__ == "__main__":
    GroqChat().run()
