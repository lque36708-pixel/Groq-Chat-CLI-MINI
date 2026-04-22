# Groq Chat Mini

A focused terminal chat client for the [Groq API](https://console.groq.com/), built around three core use cases: **chat, text summarization, and file attachment**. Supports OCR for scanned PDFs and images (Vietnamese + English).

---

## Requirements

- Python **3.10+**
- A Groq API key (free at [console.groq.com](https://console.groq.com/))
- Tesseract OCR *(only needed for scanned PDF / image OCR)*

---

## Installation

### Step 1 — Get the file

```bash
git clone https://github.com/yourname/groq-chat-mini
cd groq-chat-mini
```

Or simply download `groq_chat_mini.py` and run it directly.

### Step 2 — Install Python packages

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `requests` | Groq API calls and URL fetching |
| `rich` | Markdown rendering, tables, spinners in terminal |
| `prompt_toolkit` | Multi-line input, history, tab-completion |
| `python-docx` | Read `.docx` files |
| `openpyxl` | Read `.xlsx` spreadsheets |
| `PyMuPDF` | Read PDFs (digital and scanned) |
| `pytesseract` | OCR for scanned PDFs and images |
| `Pillow` | Image processing for the OCR pipeline |
| `rapidfuzz` | Smart file search by name (`--find-in`) |
| `pyperclip` | Clipboard support (`/copy`) — optional |

### Step 3 — Install Tesseract OCR *(optional)*

Only required if you want to attach **scanned PDFs** or **image files**. The app runs fine without it — OCR features are simply disabled.

**Linux (Debian / Ubuntu / Mint):**
```bash
sudo apt install tesseract-ocr tesseract-ocr-vie
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
1. Download the installer from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. During installation, select the **Vietnamese** language package
3. Add the Tesseract installation folder to your `PATH` environment variable

---

## Getting an API Key

Groq provides **free** API keys with generous rate limits for personal use.

1. Go to [console.groq.com](https://console.groq.com/) and sign up (Google login supported)
2. Open **API Keys** in the left sidebar
3. Click **Create API Key**, give it any name (e.g. `groq-chat-mini`)
4. Copy the key — it looks like `gsk_xxxxxxxxxxxxxxxxxxxxxxxx`

> **Important:** The key is only shown **once** immediately after creation. If you close the dialog without copying it, you'll need to create a new one.

---

## Configuring the API Key

### Option 1 — Set it inside the app *(recommended)*

Run the app, then type:

```
/set-key gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

The key is saved to the config file and loaded automatically on every launch. You won't need to set it again.

### Option 2 — Edit the config file directly

The config file is located at:
- **Linux / macOS:** `~/.groq_chat/groq_config.json`
- **Windows:** `C:\Users\<username>\.groq_chat\groq_config.json`

Create or open the file and fill in:

```json
{
    "api_key": "gsk_xxxxxxxxxxxxxxxxxxxxxxxx",
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.7,
    "system_prompt": "You are a helpful, concise AI assistant. Answer in the user's language.",
    "working_dirs": []
}
```

On Linux / macOS, restrict the file permissions to protect your key:

```bash
chmod 600 ~/.groq_chat/groq_config.json
```

---

## Running

**Linux / macOS:**
```bash
python3 groq_chat_mini.py
```

**Windows:**
```cmd
python groq_chat_mini.py
```

### Key Bindings

| Key | Action |
|---|---|
| `Ctrl+J` | Send message |
| `Enter` | New line (does not send) |
| `Ctrl+Q` | Quick exit |
| `↑ / ↓` | Browse input history |
| `Tab` | Auto-complete commands |

---

## Commands

### Conversation

| Command | Description |
|---|---|
| `/retry` | Re-send the last message (discards the previous AI response) |
| `/clear` | Clear all chat history in the current session |
| `/clear-scr` | Clear the terminal screen, keep history intact |
| `/history` | View the full conversation history as a table |
| `/search <keyword>` | Search for a keyword in the conversation history |

### File Attachment

| Command | Description |
|---|---|
| `/attach <path\|url>` | Attach a local file or webpage into the context |
| `/attach --find-in:<alias> <name>` | Find a file by name in a working directory and attach it |
| `/attach --latest-in:<alias>` | Attach the most recently modified file in a working directory |
| `/unattach` | Remove the last attached file from the context |

**Supported file formats:**

| Type | Formats |
|---|---|
| Plain text | `.txt` `.md` `.py` and most other text files |
| Office | `.docx` `.xlsx` |
| PDF | `.pdf` (digital text and scanned) |
| Images (OCR) | `.png` `.jpg` `.jpeg` `.bmp` `.gif` `.tiff` `.webp` |
| Web | Any URL starting with `http://` or `https://` |

Scanned PDFs and images are automatically OCR'd (Vietnamese + English). If a PDF already contains digital text, it is read directly without OCR.

**Examples:**
```
/attach /home/user/report.pdf
/attach C:\Users\user\Documents\report.pdf
/attach https://example.com/article
```

### Working Directories

Working directories are named shortcuts to folders you use frequently. They make `/attach --find-in` and `--latest-in` work without typing full paths.

| Command | Description |
|---|---|
| `/wd <path> <alias>` | Register a directory with a short alias (no spaces in alias) |
| `/wd` | List all registered directories |
| `/ls-wd` | List registered directories (alias for `/wd`) |
| `/wd-rm <alias>` | Remove a registered directory |

**Examples:**
```
# Linux / macOS
/wd /home/user/documents docs
/wd /home/user/projects code

# Windows — use quotes if the path contains spaces
/wd "C:\Users\user\Documents" docs
/wd "C:\Users\user\My Projects" code

# Using working directories
/attach --find-in:docs q3 report     ← finds the closest matching file by name
/attach --latest-in:code             ← attaches the most recently modified file
/wd-rm docs                          ← removes the docs alias
```

**How `--find-in` ranks results (in order of priority):**
1. Exact filename match (case-insensitive)
2. All query words present in the filename
3. Fuzzy match on filename and parent path (RapidFuzz WRatio)
4. Recency bonus — files modified recently score higher

### Display & Export

| Command | Description |
|---|---|
| `/view` | Open the last AI response in the browser (MathJax + syntax highlighting + Copy buttons) |
| `/save` | Save the conversation as a Markdown file in the current directory |
| `/export html` | Save the conversation as a styled dark-themed HTML file |
| `/copy` | Copy the last AI response to the clipboard |

### Settings

| Command | Description |
|---|---|
| `/models` | Browse the model list and select by number |
| `/model <id>` | Switch model directly by ID |
| `/temp <0.0–2.0>` | Set temperature (0 = precise, 2 = most creative) |
| `/system` | Show the current system prompt |
| `/system <text>` | Set a custom system prompt |
| `/prompt-lib` | Pick a system prompt from the built-in library |
| `/set-key <key>` | Set or update the Groq API key |
| `/info` | Show the current configuration summary |

### General

| Command | Description |
|---|---|
| `/help` | Show the full command reference |
| `/bye` | Exit |

---

## Models

Switch models with `/models` (numbered picker) or `/model <id>` (direct switch).

| Group | Model ID | Notes |
|---|---|---|
| **Deep Reasoning** | `qwen/qwen3-32b` | Academic reasoning, technical logic |
| | `openai/gpt-oss-120b` | Complex data analysis |
| | `openai/gpt-oss-20b` | High-performance reasoning |
| **Llama** | `meta-llama/llama-4-scout-17b-16e-instruct` | Instruction following |
| | `llama-3.3-70b-versatile` | Versatile, stable — **default** |
| | `llama-3.1-8b-instant` | Ultra-fast responses |
| **Specialized** | `canopylabs/orpheus-v1-english` | Text editing |
| | `allam-2-7b` | Lightweight text processing |

---

## Config & Data

All persistent data is stored at:

- **Linux / macOS:** `~/.groq_chat/`
- **Windows:** `C:\Users\<username>\.groq_chat\`

| File | Contents |
|---|---|
| `groq_config.json` | API key, model, system prompt, temperature, working dirs |
| `input_history` | Input history for up-arrow recall |

Full structure of `groq_config.json`:

```json
{
    "api_key": "gsk_...",
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.7,
    "system_prompt": "You are a helpful, concise AI assistant. Answer in the user's language.",
    "working_dirs": [
        {"path": "/home/user/documents", "name": "docs"},
        {"path": "/home/user/projects",  "name": "code"}
    ]
}
```

---

## Launching with a Keyboard Shortcut

### Linux — Cinnamon

Go to **System Settings → Keyboard → Shortcuts → Custom Shortcuts**, click **Add**, and fill in:

- **Name:** Groq Chat Mini
- **Command:** `x-terminal-emulator -e python3 /path/to/groq_chat_mini.py`

Click the new entry under **Keyboard Bindings** and press the key combination you want.

To use a specific terminal instead:
```bash
gnome-terminal -- python3 /path/to/groq_chat_mini.py
xterm -e python3 /path/to/groq_chat_mini.py
```

### Linux — GNOME

Go to **Settings → Keyboard → View and Customise Shortcuts → Custom Shortcuts** and add the same command.

### macOS

Create a file named `groq_chat_mini.command`:
```bash
#!/bin/bash
cd /path/to/
python3 groq_chat_mini.py
```
Make it executable:
```bash
chmod +x groq_chat_mini.command
```
Assign a keyboard shortcut using **Automator**, **BetterTouchTool**, or **Raycast**.

### Windows

Create a `groq_chat_mini.bat` file:
```bat
@echo off
python "C:\path\to\groq_chat_mini.py"
pause
```
Right-click the `.bat` file → **Create shortcut** → right-click the shortcut → **Properties** → set the **Shortcut key**.

---

## License

GNU General Public License (GPL) — free to use, modify, and redistribute.