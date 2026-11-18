"""ASS phrase captions with discrete current-word highlighting."""

from pathlib import Path
from typing import Sequence

from .config import RenderConfig
from .transcription import Word

STYLES = {
    "bold_yellow": ("&H00FFFFFF", "&H0000FFFF", 1, "&H90000000"),
    "clean_white": ("&H00AAAAAA", "&H00FFFFFF", 1, "&H90000000"),
    "boxed": ("&H00FFFFFF", "&H0000FFFF", 3, "&H50000000"),
}


def font_family(directory: Path) -> str:
    names = {path.stem.casefold() for path in directory.iterdir() if path.suffix.lower() in (".ttf", ".otf", ".ttc")} if directory.is_dir() else set()
    if any(name == "anton" or name.startswith("anton-") for name in names):
        return "Anton"
    if "poppins-bold" in names:
        return "Poppins"
    if "ariblk" in names or "arial-black" in names:
        return "Arial Black"
    if "nirmala" in names:
        return "Nirmala UI"
    return "Arial Black"


def display_word(text: str, uppercase: bool) -> str:
    # Neutralize ASS overrides and newline escapes in recognized text.
    text = " ".join(text.split()).replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    return text.upper() if uppercase else text


def phrase_groups(words: Sequence[Word], config: RenderConfig) -> list[list[Word]]:
    groups: list[list[Word]] = []
    width_budget = (1080 - 2 * config.margin_horizontal) / (config.font_size * 0.65)
    for word in words:
        text = display_word(word.text, config.uppercase)
        previous = groups[-1] if groups else []
        length = len(" ".join(display_word(item.text, config.uppercase) for item in [*previous, word]))
        if (not previous or len(previous) >= config.max_words or length > width_budget
            or word.start - previous[-1].end > 0.8 or previous[-1].text.endswith((".", "!", "?"))):
            groups.append([word])
        else:
            previous.append(word)
    return groups


def ass_time(seconds: float) -> str:
    ticks = max(0, round(seconds * 100))
    hours, ticks = divmod(ticks, 360000)
    minutes, ticks = divmod(ticks, 6000)
    whole, hundredths = divmod(ticks, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{hundredths:02d}"


def generate_ass(words: Sequence[Word], config: RenderConfig, family: str, duration: float = 60.0) -> str:
    # If any recognized word contains Indic / Devanagari characters, automatically
    # use Nirmala UI to prevent missing glyphs / tofu boxes in libass rendering.
    all_text = " ".join(w.text for w in words)
    if any("\u0900" <= ch <= "\u0d7f" for ch in all_text):
        family = "Nirmala UI"

    base, highlight, border, back = STYLES[config.caption_style]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{family},{config.font_size},{base},{highlight},&H00000000,{back},-1,0,0,0,100,100,0,0,{border},{config.outline},{config.shadow},2,{config.margin_horizontal},{config.margin_horizontal},{config.margin_bottom},1
Style: Credit,{family},36,&H00FFFFFF,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,2,1,8,40,40,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    if config.creator_credit and config.creator_credit.strip():
        tag = config.creator_credit.strip()
        if not tag.startswith("@"):
            tag = f"@{tag}"
        credit_end = ass_time(duration)
        lines.append(f"Dialogue: 1,0:00:00.00,{credit_end},Credit,,0,0,0,,{tag}\n")

    base_bgr, highlight_bgr = f"&H{base[-6:]}&", f"&H{highlight[-6:]}&"
    groups = phrase_groups(words, config)
    for g_idx, group in enumerate(groups):
        phrase = [display_word(word.text, config.uppercase) for word in group]
        # Fit unusually long individual words rather than overflowing safe margins.
        font_size = min(config.font_size, int((1080 - 2 * config.margin_horizontal) / (max(1, len(" ".join(phrase))) * 0.65)))
        cursor = group[0].start
        for index, word in enumerate(group):
            w_start = max(cursor, word.start)
            if index + 1 < len(group):
                w_end = max(w_start + 0.12, group[index + 1].start)
            else:
                # Hold completed phrase until next phrase begins (or up to 0.75s sustain)
                if g_idx + 1 < len(groups):
                    next_start = groups[g_idx + 1][0].start
                    w_end = max(word.end, min(next_start, word.end + 0.75))
                else:
                    w_end = max(word.end, w_start + 0.6)
            cursor = w_end

            if round(w_end * 100) <= round(w_start * 100):
                continue
            text = " ".join(f"{{\\1c{highlight_bgr}}}{token}{{\\1c{base_bgr}}}" if at == index else token
                            for at, token in enumerate(phrase))
            lines.append(f"Dialogue: 0,{ass_time(w_start)},{ass_time(w_end)},Caption,,0,0,0,,{{\\fs{font_size}}}{text}\n")
    return "".join(lines)
