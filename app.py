import os
import json
import shutil
import hashlib
import threading
from datetime import datetime
import torch
import opencc
import gradio as gr
from google import genai

# ----------------- SYSTEM PRIORITY & THREADING -----------------
try:
    os.nice(15)
except Exception:
    pass

CPU_THREADS = int(os.environ.get("CPU_THREADS", 4))
torch.set_num_threads(CPU_THREADS)

# ----------------- CONFIGURATION & STORAGE -----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

DATA_DIR = "/data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

models = {
    "whisper": None,
    "funasr": None
}

model_locks = {
    "whisper": threading.Lock(),
    "funasr": threading.Lock()
}

# Pre-instantiate OpenCC converters globally
OPENCC_S2T = opencc.OpenCC("s2t")
OPENCC_T2S = opencc.OpenCC("t2s")

def get_lang_code(lang: str) -> str:
    mapping = {
        "Cantonese": "yue",
        "Mandarin": "zh",
        "English": "en"
    }
    return mapping.get(lang, lang.lower())

def apply_opencc(text: str, zh_format: str, lang: str = "Cantonese") -> str:
    if not text or lang == "English":
        return text
    if zh_format == "Traditional Chinese":
        return OPENCC_S2T.convert(text)
    elif zh_format == "Simplified Chinese":
        return OPENCC_T2S.convert(text)
    return text

# ----------------- PERSISTENCE HELPERS -----------------
def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_record_path(md5_hash: str, lang: str) -> str:
    lang_code = get_lang_code(lang)
    return os.path.join(RESULTS_DIR, f"{md5_hash}_{lang_code}.json")

def find_stored_audio(md5_hash: str) -> str | None:
    """Finds the stored audio file regardless of extension."""
    if not md5_hash:
        return None
    for ext in ('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus', '.webm'):
        path = os.path.join(AUDIO_DIR, f"{md5_hash}{ext}")
        if os.path.exists(path):
            return path
    return None

def load_record(md5_hash: str, lang: str) -> dict | None:
    if not md5_hash:
        return None
    path = get_record_path(md5_hash, lang)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_record(md5_hash: str, lang: str, data: dict):
    if not md5_hash:
        return
    existing = load_record(md5_hash, lang) or {}
    existing.update(data)
    existing["lang"] = lang
    existing["updated_at"] = datetime.now().isoformat()
    with open(get_record_path(md5_hash, lang), "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

# ----------------- MODEL LOADING -----------------
def monkeypatch_av():
    try:
        import av
        if not hasattr(av, "_patched_for_whisper"):
            _orig_av_open = av.open
            def _safe_av_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return _orig_av_open(*args, **kwargs)
            av.open = _safe_av_open
            av._patched_for_whisper = True
    except ImportError:
        pass

def load_whisper_model():
    if models["whisper"] is None:
        monkeypatch_av()
        from faster_whisper import WhisperModel
        models["whisper"] = WhisperModel(
            "medium.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=CPU_THREADS
        )
    return "✅ English model (`faster-whisper medium.en`, int8 CPU) is ready."

def load_funasr_model():
    if models["funasr"] is None:
        from funasr import AutoModel
        models["funasr"] = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            device="cpu",
            ncpu=CPU_THREADS,
            trust_remote_code=True
        )
    return "✅ Chinese model (`SenseVoiceSmall` + `ct-punc`, CPU) is ready."

def load_model(lang: str):
    if lang == "English":
        with model_locks["whisper"]:
            return load_whisper_model()
    else:
        with model_locks["funasr"]:
            return load_funasr_model()

# ----------------- CACHE & PROCESSING LOGIC -----------------
def check_md5_cache(md5_str: str, lang: str, zh_format: str, cantonese_to_written: bool):
    """Checks if an MD5 hash exists on the server for the chosen language."""
    if not md5_str or "|" not in md5_str:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    md5_hash, filename = md5_str.split("|", 1)
    record = load_record(md5_hash, lang)
    existing_audio = find_stored_audio(md5_hash)

    if record and record.get("raw_text"):
        audio_val = existing_audio if (existing_audio and os.path.exists(existing_audio)) else gr.update()
        raw_text_unconverted = record.get("raw_text", "")

        if lang == "Cantonese":
            target_key = "formatted_written" if cantonese_to_written else "formatted_spoken"
            formatted_raw = record.get(target_key) or record.get("formatted_text", "")
        else:
            formatted_raw = record.get("formatted_general") or record.get("formatted_text", "")

        display_raw = apply_opencc(raw_text_unconverted, zh_format, lang)
        display_formatted = apply_opencc(formatted_raw, zh_format, lang)

        status_msg = (
            f"⚡ **Cache Hit ({lang})!** `{filename}` (`{md5_hash[:8]}...`) was processed on "
            f"`{record.get('updated_at', 'previously')[:19]}`. Upload skipped and results loaded directly. "
            f"Click **Start Transcription** or **Start AI Refinement** to re-run."
        )
        return (
            status_msg,
            display_raw,
            display_formatted,
            audio_val,
            md5_hash
        )

    # Audio file exists on server, but not yet transcribed for this language
    if existing_audio:
        return (
            f"⚡ Audio file `{filename}` is already on server, but no `{lang}` record exists. "
            f"Upload skipped. Click **Start Transcription** to process in {lang}.",
            "",
            "",
            existing_audio,
            md5_hash
        )

    return (
        f"📤 New file `{filename}` (`{md5_hash[:8]}...`). Uploading to server...",
        gr.update(),
        gr.update(),
        gr.update(),
        md5_hash
    )

def process_transcription(audio_upload, current_md5, lang, zh_format, en_prompt):
    audio_path = None
    if audio_upload and os.path.exists(audio_upload):
        audio_path = audio_upload
    elif current_md5:
        stored = find_stored_audio(current_md5)
        if stored:
            audio_path = stored

    if not audio_path:
        raise gr.Error("Please upload an audio file first.")

    md5_hash = compute_file_md5(audio_path)
    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    persistent_audio_path = os.path.join(AUDIO_DIR, f"{md5_hash}{ext}")

    if not os.path.exists(persistent_audio_path):
        shutil.copy2(audio_path, persistent_audio_path)

    gr.Info(f"Transcription ({lang}) started on ARM CPU. This may take a moment...")

    if lang == "English":
        load_model("English")
        model = models["whisper"]
        segments, _ = model.transcribe(
            persistent_audio_path,
            initial_prompt=en_prompt if en_prompt else None
        )
        raw_text = "".join([segment.text for segment in segments])
    else:
        load_model(lang)
        model = models["funasr"]
        lang_code = "yue" if lang == "Cantonese" else "zh"

        res = model.generate(input=persistent_audio_path, language=lang_code, merge_vad=False)
        raw_text = res[0]["text"] if isinstance(res, list) and res else str(res)

    # Save to language-specific JSON file without OpenCC
    save_record(md5_hash, lang, {
        "md5": md5_hash,
        "audio_path": persistent_audio_path,
        "raw_text": raw_text
    })

    display_raw = apply_opencc(raw_text, zh_format, lang)
    lang_code = get_lang_code(lang)
    return display_raw, md5_hash, f"✅ {lang} transcription completed and saved to `{md5_hash[:8]}_{lang_code}.json`."

def process_refinement(raw_text, current_md5, lang, zh_format, cantonese_to_written):
    if not raw_text or not raw_text.strip():
        raise gr.Error("No transcription found. Please transcribe an audio file first.")

    if not GEMINI_API_KEY:
        raise gr.Error("GEMINI_API_KEY environment variable is not set.")

    # Read original un-converted source text if available to prevent translation artifacts
    source_text = raw_text
    if current_md5:
        record = load_record(current_md5, lang)
        if record and record.get("raw_text"):
            source_text = record["raw_text"]

    client = genai.Client(api_key=GEMINI_API_KEY)

    if lang == "Cantonese" and cantonese_to_written:
        prompt = f"""You are a professional editor specializing in translating spoken Cantonese into formal Standard Written Chinese (現代標準漢語 / 規範書面語).

Your task is to convert the following raw spoken Cantonese transcription into polished, natural Standard Written Chinese.

### ABSOLUTE CONVERSION RULES:
1. **ELIMINATE ALL CANTONESE DIALECT & COLLOQUIALISMS:**
   The output must read as natural, professional written Chinese. It must NOT contain Cantonese colloquial words or particles:
   - 嘅 -> 的
   - 咗 -> 了
   - 喺 / 喺度 -> 在 / 正在
   - 唔 / 咪 -> 不 / 不要
   - 係 -> 是
   - 佢 / 佢哋 -> 他（她/它） / 他們
   - 我哋 / 你哋 -> 我們 / 你們
   - 乜嘢 / 點解 / 邊度 / 幾時 -> 什麼 / 為什麼 / 哪裡 / 何時
   - 諗 / 睇 / 講 / 搵 / 畀 -> 想 / 看 / 說 / 找 / 給
   - 冇 -> 沒有
   - 仲 -> 還
   - 呢個 / 嗰個 -> 這個 / 那個
   - 緊 (progressive marker) -> 正在 / 著
   - Modal particles (喇, 喎, 囉, 㗎, 嘛, 喱, 呀) -> Remove or replace with standard punctuation or standard particles (了, 呢, 吧).

2. **ZERO INFORMATION LOSS (CRITICAL):**
   - Translate sentence-by-sentence. **DO NOT summarize, condense, or omit ANY content, nuance, or detail.**
   - Preserve all proper nouns, technical terms, specific numbers, and original logical flow verbatim.

3. **FORMATTING:**
   - Organize into logical paragraphs.
   - Add appropriate Markdown headings (`##`, `###`) where topics change.
   - Fix transcription typos and add accurate punctuation.

Output ONLY the converted, formatted text. Do not provide greetings, explanations, or commentary.

Raw Spoken Cantonese Transcription:
{source_text}
"""
    else:
        prompt = f"""You are an expert transcriber and editor.
Your task is to tidy up the following raw transcription.

CRITICAL RULES:
1. ZERO INFORMATION LOSS: DO NOT summarize, omit, or lose ANY content from the original audio. Keep all details, terms, and sentences intact.
2. FORMATTING ONLY:
   - Add logical paragraph breaks.
   - Fix missing or incorrect punctuation.
   - Correct obvious transcription mistakes and typos.
   - Add appropriate Markdown headings (##, ###) to structure the document.

The language of the text is {lang}.
Output ONLY the refined transcription without any introductory or concluding comments.

Raw Transcription:
{source_text}
"""

    gr.Info(f"AI Refinement ({lang}) in progress...")
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    formatted_raw = response.text

    # Store in language-isolated JSON record
    if current_md5:
        update_data = {}
        if lang == "Cantonese":
            if cantonese_to_written:
                update_data["formatted_written"] = formatted_raw
            else:
                update_data["formatted_spoken"] = formatted_raw
        else:
            update_data["formatted_general"] = formatted_raw
        save_record(current_md5, lang, update_data)

    return apply_opencc(formatted_raw, zh_format, lang)

def auto_trigger_refinement(auto_start, raw_text, current_md5, lang, zh_format, cantonese_to_written):
    if auto_start and raw_text:
        return process_refinement(raw_text, current_md5, lang, zh_format, cantonese_to_written)
    return gr.update()

# ----------------- DYNAMIC UI HANDLERS -----------------
def on_lang_dropdown_change(new_lang, current_md5, zh_format, cantonese_to_written):
    """
    Adjusts UI visibility and automatically loads existing records for the new language
    if the audio is already loaded/cached.
    """
    # Visibility updates
    if new_lang == "English":
        zh_fmt_up = gr.update(visible=False)
        en_prompt_up = gr.update(visible=True)
        cant_up = gr.update(visible=False)
    elif new_lang == "Cantonese":
        zh_fmt_up = gr.update(visible=True)
        en_prompt_up = gr.update(visible=False)
        cant_up = gr.update(visible=True)
    else:  # Mandarin
        zh_fmt_up = gr.update(visible=True)
        en_prompt_up = gr.update(visible=False)
        cant_up = gr.update(visible=False)

    if not current_md5:
        return zh_fmt_up, en_prompt_up, cant_up, gr.update(), gr.update(), gr.update()

    # Load existing language file if available
    record = load_record(current_md5, new_lang)
    lang_code = get_lang_code(new_lang)

    if record and record.get("raw_text"):
        raw_text = record.get("raw_text", "")
        if new_lang == "Cantonese":
            target_key = "formatted_written" if cantonese_to_written else "formatted_spoken"
            fmt_raw = record.get(target_key) or record.get("formatted_text", "")
        else:
            fmt_raw = record.get("formatted_general") or record.get("formatted_text", "")

        disp_raw = apply_opencc(raw_text, zh_format, new_lang)
        disp_fmt = apply_opencc(fmt_raw, zh_format, new_lang)
        status = f"⚡ Loaded cached `{new_lang}` record (`{current_md5[:8]}_{lang_code}.json`)."
        return zh_fmt_up, en_prompt_up, cant_up, disp_raw, disp_fmt, status

    status = f"ℹ️ Audio `{current_md5[:8]}...` has no `{new_lang}` record yet. Click Start Transcription to generate it."
    return zh_fmt_up, en_prompt_up, cant_up, "", "", status

def on_zh_format_change(raw_val, formatted_val, current_md5, lang, zh_format, cantonese_to_written):
    """Re-renders existing text into target Chinese variant from source data."""
    if current_md5:
        record = load_record(current_md5, lang)
        if record and record.get("raw_text"):
            raw_base = record.get("raw_text", "")
            if lang == "Cantonese":
                target_key = "formatted_written" if cantonese_to_written else "formatted_spoken"
                fmt_base = record.get(target_key) or record.get("formatted_text", "")
            else:
                fmt_base = record.get("formatted_general") or record.get("formatted_text", "")
            return apply_opencc(raw_base, zh_format, lang), apply_opencc(fmt_base, zh_format, lang)

    return apply_opencc(raw_val, zh_format, lang), apply_opencc(formatted_val, zh_format, lang)

def on_cantonese_to_written_change(current_md5, lang, zh_format, cantonese_to_written):
    """Switches immediately between written and spoken formatted versions."""
    if not current_md5 or lang != "Cantonese":
        return gr.update()
    record = load_record(current_md5, "Cantonese")
    if not record:
        return gr.update()

    target_key = "formatted_written" if cantonese_to_written else "formatted_spoken"
    formatted = record.get(target_key)
    if formatted:
        return apply_opencc(formatted, zh_format, lang)
    return gr.update()

# ----------------- CLIENT-SIDE JAVASCRIPT -----------------
HEAD_HTML = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/spark-md5/3.0.2/spark-md5.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    let bypassedFile = null;

    function computeMD5(file) {
        return new Promise((resolve, reject) => {
            const chunkSize = 2097152;
            const spark = new SparkMD5.ArrayBuffer();
            const fileReader = new FileReader();
            let cursor = 0;

            fileReader.onload = function(e) {
                spark.append(e.target.result);
                cursor += chunkSize;
                if (cursor < file.size) {
                    loadNext();
                } else {
                    resolve(spark.end());
                }
            };
            fileReader.onerror = () => reject('MD5 read failed');

            function loadNext() {
                const slice = file.slice(cursor, Math.min(cursor + chunkSize, file.size));
                fileReader.readAsArrayBuffer(slice);
            }
            loadNext();
        });
    }

    document.addEventListener('change', async (e) => {
        const input = e.target;
        if (input.tagName === 'INPUT' && input.type === 'file' && input.closest('#audio_upload_box')) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];

            if (bypassedFile === file) {
                bypassedFile = null;
                return;
            }

            e.stopPropagation();
            e.stopImmediatePropagation();

            const md5 = await computeMD5(file);

            const bridgeInput = document.querySelector('#md5_bridge textarea, #md5_bridge input');
            if (bridgeInput) {
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set 
                            || Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
                setter.call(bridgeInput, md5 + "|" + file.name);
                bridgeInput.dispatchEvent(new Event('input', { bubbles: true }));
            }

            await new Promise(r => setTimeout(r, 450));
            const statusText = document.querySelector('#cache_status')?.innerText || "";

            if (statusText.includes("Upload skipped") || statusText.includes("Cache Hit")) {
                input.value = "";
            } else {
                bypassedFile = file;
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }, true);
});
</script>
"""

js_download_raw = """
() => {
    const el = document.querySelector('#raw_output textarea');
    if (!el || !el.value.trim()) {
        alert('No transcription text to download.');
        return;
    }
    const blob = new Blob([el.value], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'raw_transcription.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
"""

js_download_formatted = """
() => {
    const el = document.querySelector('#formatted_output textarea');
    if (!el || !el.value.trim()) {
        alert('No formatted text to download.');
        return;
    }
    const blob = new Blob([el.value], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'formatted_transcription.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
"""

# ----------------- GRADIO UI -----------------
with gr.Blocks(title="AI Audio Transcription & Refinement", head=HEAD_HTML) as app:
    current_md5 = gr.State("")
    md5_bridge = gr.Textbox(elem_id="md5_bridge", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Section 1: Language settings")
            lang_dropdown = gr.Radio(["Cantonese", "Mandarin", "English"], label="Language", value="Cantonese")
            zh_format = gr.Radio(["Traditional Chinese", "Simplified Chinese"], label="Chinese Output Format", value="Traditional Chinese", visible=True)
            en_prompt = gr.Textbox(label="Initial Prompt", placeholder="(Optional) Enter initial prompt for faster-whisper...", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Section 2: Input audio")
            cache_status = gr.Markdown("Select or drop an audio file. MD5 will be checked before uploading.", elem_id="cache_status")
            audio_upload = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload Audio",
                elem_id="audio_upload_box"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Section 3: Raw Transcription")
            load_status = gr.Markdown("Model status: *Ready to load on demand*")
            transcribe_btn = gr.Button("Start Transcription", variant="primary")
            raw_output = gr.Textbox(label="Raw Transcription Output", lines=8, elem_id="raw_output")
            download_raw_btn = gr.Button("⬇️ Download Raw Transcription (.txt)")
            download_raw_btn.click(fn=None, js=js_download_raw)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Section 4: AI Refinement (Gemini 3.5 Flash Lite)")
            auto_refine = gr.Checkbox(label="Auto start AI refinement once transcription is done", value=True)
            cantonese_to_written = gr.Checkbox(label="粵語口語轉書面語", value=True, visible=True)
            refine_btn = gr.Button("Start AI Refinement Manually")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Section 5: Formatted Transcription")
            formatted_output = gr.Textbox(label="Formatted Output", lines=12, elem_id="formatted_output")
            download_formatted_btn = gr.Button("⬇️ Download Formatted Transcription (.txt)")
            download_formatted_btn.click(fn=None, js=js_download_formatted)

    # ----------------- UI & EVENT BINDINGS -----------------
    # Language change: adjusts UI controls and swaps rendered content if file already loaded
    lang_dropdown.change(
        fn=on_lang_dropdown_change,
        inputs=[lang_dropdown, current_md5, zh_format, cantonese_to_written],
        outputs=[zh_format, en_prompt, cantonese_to_written, raw_output, formatted_output, cache_status],
        concurrency_limit=None
    )

    # Simplified / Traditional conversion toggle
    zh_format.change(
        fn=on_zh_format_change,
        inputs=[raw_output, formatted_output, current_md5, lang_dropdown, zh_format, cantonese_to_written],
        outputs=[raw_output, formatted_output],
        concurrency_limit=None
    )

    # Toggle between Cantonese written and spoken refinement versions
    cantonese_to_written.change(
        fn=on_cantonese_to_written_change,
        inputs=[current_md5, lang_dropdown, zh_format, cantonese_to_written],
        outputs=formatted_output,
        concurrency_limit=None
    )

    # MD5 Cache check runs immediately
    md5_bridge.change(
        fn=check_md5_cache,
        inputs=[md5_bridge, lang_dropdown, zh_format, cantonese_to_written],
        outputs=[cache_status, raw_output, formatted_output, audio_upload, current_md5],
        concurrency_limit=None
    )

    # Pre-load model on upload completion
    audio_upload.upload(
        fn=load_model,
        inputs=lang_dropdown,
        outputs=load_status,
        concurrency_id="transcribe_queue",
        concurrency_limit=1
    )

    # 1. TRANSCRIPTION QUEUE: Strictly 1 at a time (CPU bound)
    transcribe_event = transcribe_btn.click(
        fn=process_transcription,
        inputs=[audio_upload, current_md5, lang_dropdown, zh_format, en_prompt],
        outputs=[raw_output, current_md5, cache_status],
        concurrency_id="transcribe_queue",
        concurrency_limit=1
    )

    # 2. REFINEMENT QUEUE: Runs in parallel (I/O bound)
    transcribe_event.success(
        fn=auto_trigger_refinement,
        inputs=[auto_refine, raw_output, current_md5, lang_dropdown, zh_format, cantonese_to_written],
        outputs=formatted_output,
        concurrency_id="refine_pool",
        concurrency_limit=10
    )

    refine_btn.click(
        fn=process_refinement,
        inputs=[raw_output, current_md5, lang_dropdown, zh_format, cantonese_to_written],
        outputs=formatted_output,
        concurrency_id="refine_pool",
        concurrency_limit=10
    )

app.queue(default_concurrency_limit=4)
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    allowed_paths=[AUDIO_DIR]
)
