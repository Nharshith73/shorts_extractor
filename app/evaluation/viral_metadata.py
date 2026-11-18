"""Attention-Grabbing Viral Metadata System for YouTube Shorts, Reels, and TikTok.

Engineers high-CTR click-worthy titles with curiosity gaps, natural story-driven
descriptions with streamer credits and engagement CTAs, and algorithmically
targeted tags for maximum reach on YouTube Shorts feeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import random
import re
from typing import Any

LOGGER = logging.getLogger(__name__)

# Curated High-CTR Viral Title Templates by Archetype
VIRAL_TITLE_TEMPLATES = {
    "shock": [
        "{creator} DID NOT EXPECT THIS TO HAPPEN 💀 #Shorts",
        "WAIT TILL YOU SEE WHAT HAPPENED... 😱 #Shorts",
        "NO WAY HE ACTUALLY DID THIS?! 💀 #Shorts",
        "THE ENTIRE CHAT FROZE AFTER THIS... 😱 #Shorts",
        "EVERYBODY WAS SPEECHLESS... 😭 #Shorts",
        "THIS HAS TO BE SCRIPTED... NO WAY! 🔥 #Shorts",
        "HE REALLY THOUGHT HE WAS SAFE 😂💀 #Shorts",
    ],
    "funny": [
        "HE INSTANTLY REGRETTED THIS DECISION 💀 #Shorts",
        "BRO EXPERIENCED PURE PAIN 😂😭 #Shorts",
        "HOW DID HE NOT SEE THAT COMING?! 💀 #Shorts",
        "HE COULD NOT STOP LAUGHING 😂💀 #Shorts",
        "THE DISRESPECT WAS UNREAL 😭💀 #Shorts",
        "BRO THOUGHT HE WAS HIM 😂 #Shorts",
        "WHEN THE JOKE GOES TOO FAR 💀 #Shorts",
    ],
    "clutch": [
        "THE MOST INSANE CLUTCH YOU'LL SEE TODAY 🔥 #Shorts",
        "HE ACTUALLY PULLED IT OFF?! 😱 #Shorts",
        "BRO TURNED INTO A GOD ⚡ #Shorts",
        "UNREAL CLUTCH MOMENT ON STREAM 🔥 #Shorts",
        "1 IN A MILLION MOMENT RIGHT HERE 🎯 #Shorts",
    ],
    "chat": [
        "{creator}'s CHAT WENT ABSOLUTELY CRAZY! 😱 #Shorts",
        "CHAT WAS SPAMMING W's IN THE CHAT! 🔥 #Shorts",
        "THE WHOLE STREAM LOST THEIR MINDS 😭 #Shorts",
        "BRO BROKE THE STREAM WITH THIS ⚡ #Shorts",
        "WHEN CHAT PREDICTS THE FUTURE 💀 #Shorts",
    ],
    "general": [
        "YOU WON'T BELIEVE WHAT HAPPENED NEXT 😱 #Shorts",
        "WAIT FOR THE END... 😭💀 #Shorts",
        "NEVER DO THIS ON LIVE STREAM 💀 #Shorts",
        "IS THIS THE CRAZIEST MOMENT OF THE YEAR?! 🔥 #Shorts",
        "BRO REALLY DID THIS ON CAMERA 💀 #Shorts",
    ],
}

ON_SCREEN_HOOKS = {
    "shock": ["WAIT FOR IT... 😱", "NO WAY HE DID THIS 💀", "WHAT JUST HAPPENED?!", "BRO FROZE UP 😭"],
    "funny": ["BRO REALLY THOUGHT 😂", "INSTANT REGRET 💀", "HOW DID THIS HAPPEN 😭", "I'M CRYING 😂💀"],
    "clutch": ["WATCH THIS CLUTCH 🔥", "ABSOLUTE GOD MODE ⚡", "HOW DID HE HIT THAT?! 🎯", "UNREAL REACTION 😱"],
    "chat": ["CHAT WAS NOT READY 😭", "THE STREAM WENT WILD 🔥", "EVERYONE SPAM W 💬", "PURE CHAOS 💀"],
    "general": ["DON'T BLINK... ⚡", "WATCH TILL THE END 😭", "IS THIS REAL?! 😱", "BRO HAD NO IDEA 💀"],
}

CALLS_TO_ACTION = [
    "Did you see this coming? Drop a comment below! 👇",
    "Rate this moment 1-10 in the comments! 💬",
    "What would YOU do in this situation? Let me know below! 👇",
    "Type 'W' in the comments if you caught this live! 🔥",
    "Tag a friend who would totally do this! 😂",
]

# Natural, human, high-energy storytelling descriptions (NO TELEMETRY / NO ROBOT TEXT)
STORY_TEMPLATES = {
    "clutch": [
        "Unreal clutch play caught live on stream with @{creator}! Watch till the end to see how this play unfolded.",
        "Absolute god-mode moment from @{creator}! Nobody in the chat expected this to actually work out.",
        "One of the craziest plays of the entire stream! @{creator} was completely locked in for this one.",
    ],
    "funny": [
        "Absolute comedy moment from @{creator}'s live stream! Nobody in the chat could stop laughing at what just happened.",
        "You won't believe what happened live on @{creator}'s stream. Pure instant regret caught on camera!",
        "When the joke goes way too far on stream! @{creator}'s reaction is honestly priceless.",
    ],
    "shock": [
        "You won't believe what happened live on @{creator}'s stream! The whole audience was in complete disbelief.",
        "Nobody saw this coming on @{creator}'s stream! Watch till the very end to see the full reaction.",
        "An unforgettable moment from @{creator}'s live stream. The timing could not have been more perfect!",
    ],
    "chat": [
        "The chat completely lost it during this wild moment on @{creator}'s stream! The hype was unmatched.",
        "Everyone in the chat started spamming W's during this unreal moment with @{creator}!",
        "The stream went into absolute chaos when this happened on @{creator}'s stream. Watch the reaction!",
    ],
    "general": [
        "An unreal moment caught live on stream with @{creator}! Watch till the end to see the full reaction.",
        "One of the best highlights from @{creator}'s live broadcast today. Pure entertainment from start to finish!",
        "You have to see this moment from @{creator}'s stream! Things got chaotic real fast.",
    ],
}


def clean_text_for_metadata(text: str) -> str:
    """Strip telemetry, robot text, duplicate headers, dividers, and hashtags from input text."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove telemetry patterns like "Kick chat surged 2.7x baseline (148 messages in 10s)"
        if re.search(r"[A-Za-z]+ chat surged [\d\.]+x baseline", line) or re.search(r"\d+ messages in \d+s", line):
            continue
        # Remove repeated labels
        if line.startswith("Spoken dialogue:") or "Spoken dialogue:" in line:
            clean_part = re.sub(r"^Spoken dialogue:\s*", "", line).strip(' "')
            if clean_part and not re.search(r"chat surged|baseline", clean_part):
                lines.append(clean_part)
            continue
        # Remove existing template scaffolding
        if "━━━" in line or line.startswith("📺 Streamer:") or line.startswith("🌟 Daily") or line.startswith("👉 LIKE") or line.startswith("💬"):
            continue
        if line.startswith("#Shorts") or line.startswith("#shorts"):
            continue
        if any(h in line for h in ["🔥💀", "PURE CHAOS", "WHAT JUST HAPPENED", "BRO REALLY THOUGHT", "WATCH THIS CLUTCH"]):
            continue
        lines.append(line)
    return " ".join(lines).strip()


@dataclass
class ViralMetadata:
    """Complete attention-grabbing packaging for a vertical Short."""
    title: str
    suggested_hook: str
    description: str
    tags: list[str]
    category: str
    call_to_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ViralMetadataGenerator:
    """Produces viral-engineered titles, on-screen hooks, rich descriptions, and SEO tags."""

    @classmethod
    def generate(
        cls,
        dialogue: str = "",
        creator: str = "",
        platform: str = "youtube",
        category: str = "funny",
        spike_ratio: float = 0.0,
        raw_summary: str = "",
        use_ai: bool = True,
    ) -> ViralMetadata:
        """Generate high-performing viral metadata using AI when available with instant rule fallback."""
        creator_clean = creator.lstrip("@").strip() if creator else "Streamer"
        platform_clean = platform.lower().strip()

        # Sanitize any incoming dialogue and summary to purge robot/telemetry text
        clean_dia = clean_text_for_metadata(dialogue)
        clean_sum = clean_text_for_metadata(raw_summary)

        # 1. Try Groq AI generation if available and requested
        if use_ai:
            ai_meta = cls._try_groq_generation(
                dialogue=clean_dia,
                creator=creator_clean,
                platform=platform_clean,
                category=category,
                spike_ratio=spike_ratio,
                raw_summary=clean_sum,
            )
            if ai_meta:
                return ai_meta

        # 2. Algorithmic High-CTR Rule Engine (Zero-Cost / Offline Guarantee)
        return cls._generate_algorithmic(
            dialogue=clean_dia,
            creator=creator_clean,
            platform=platform_clean,
            category=category,
            spike_ratio=spike_ratio,
            raw_summary=clean_sum,
        )

    @classmethod
    def _try_groq_generation(
        cls,
        dialogue: str,
        creator: str,
        platform: str,
        category: str,
        spike_ratio: float,
        raw_summary: str,
    ) -> ViralMetadata | None:
        """Call Groq LLM with specialized viral YouTube Shorts prompt."""
        try:
            from app.evaluation.providers.groq import GroqHighlightJudgeProvider
            provider = GroqHighlightJudgeProvider()
            if not provider.has_valid_key():
                return None

            system_prompt = (
                "You are an elite YouTube Shorts and TikTok viral content strategist.\n"
                "Your objective is to craft an attention-grabbing, high-CTR package for a gaming/stream highlight.\n"
                "CRITICAL RULES:\n"
                "1. 'title': Must be under 80 characters, highly click-worthy with strong curiosity gaps, "
                "personality, and 1-2 emojis (e.g. 💀, 😭, 😱, 🔥, 😂). MUST END with '#Shorts'. Never use outer quotation marks. Never write boring summaries.\n"
                "2. 'suggested_hook': 3-5 punchy words in ALL CAPS with an emoji for on-screen caption text (e.g. 'BRO REALLY THOUGHT 💀').\n"
                "3. 'description': Engaging, natural 2-sentence story summary written in entertaining streamer language. "
                "NEVER include mechanical chat telemetry like 'chat surged 2.7x baseline' or 'messages in 10s'. Focus on the human story and viewer excitement.\n"
                "4. 'tags': Array of 15 to 20 optimized, search-friendly tags for YouTube Shorts algorithm.\n"
                "5. 'call_to_action': 1-sentence spicy question to pin in the comments to drive replies.\n"
                "Return strictly valid JSON matching this schema:\n"
                '{"title": "...", "suggested_hook": "...", "description": "...", "tags": ["..."], "call_to_action": "..."}'
            )

            prompt_user = (
                f"Streamer: {creator} (Platform: {platform.upper()})\n"
                f"Event Category: {category}\n"
                f"Action Context: {raw_summary or 'Exciting live stream moment'}\n"
                f"Dialogue: '{dialogue[:300]}'\n\n"
                "Generate the viral YouTube Shorts package JSON now:"
            )

            raw_json, _ = provider._call_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0.7,
                timeout_seconds=12.0,
            )

            data = json.loads(raw_json)
            title = data.get("title", "").strip().strip('"\'')
            if "#Shorts" not in title and "#shorts" not in title:
                title = f"{title} #Shorts"

            hook = data.get("suggested_hook", "WHAT JUST HAPPENED?! 💀").strip().upper()
            cta = data.get("call_to_action", random.choice(CALLS_TO_ACTION)).strip()
            raw_desc = clean_text_for_metadata(data.get("description", raw_summary or "").strip())
            if not raw_desc or "chat surged" in raw_desc:
                raw_desc = random.choice(STORY_TEMPLATES.get(category, STORY_TEMPLATES["general"])).format(creator=creator)

            tags = data.get("tags") or []
            if not isinstance(tags, list) or len(tags) < 5:
                tags = cls._build_tags(creator, platform, category, dialogue)

            full_desc = cls._format_structured_description(
                hook=hook,
                summary=raw_desc,
                creator=creator,
                platform=platform,
                cta=cta,
                tags=tags,
            )

            return ViralMetadata(
                title=title[:100],
                suggested_hook=hook,
                description=full_desc,
                tags=tags,
                category=category,
                call_to_action=cta,
            )
        except Exception as exc:
            LOGGER.debug("Viral Groq AI generation fallback: %s", exc)
            return None

    @classmethod
    def _generate_algorithmic(
        cls,
        dialogue: str,
        creator: str,
        platform: str,
        category: str,
        spike_ratio: float,
        raw_summary: str,
    ) -> ViralMetadata:
        """High-energy heuristic engine generating viral package instantly with zero telemetry."""
        detected_archetype = cls._detect_archetype(dialogue, category, spike_ratio)

        # 1. Title Generation
        templates = VIRAL_TITLE_TEMPLATES.get(detected_archetype, VIRAL_TITLE_TEMPLATES["general"])
        template = random.choice(templates)
        title = template.format(creator=creator)

        # If dialogue contains a punchy short quote, prioritize high-engagement quote title without outer quotes
        quote_match = cls._extract_punchy_quote(dialogue)
        if quote_match and len(quote_match) <= 35:
            clean_q = quote_match.strip(' "“’\'')
            if clean_q and not any(t in clean_q.lower() for t in ("chat surged", "baseline", "messages in")):
                title = f"{clean_q.upper()} 💀 #Shorts"

        # 2. Hook Generation
        hook_options = ON_SCREEN_HOOKS.get(detected_archetype, ON_SCREEN_HOOKS["general"])
        hook = random.choice(hook_options)

        # 3. Call to Action
        cta = random.choice(CALLS_TO_ACTION)

        # 4. Tags
        tags = cls._build_tags(creator, platform, category, dialogue)

        # 5. Formatted Natural Story Description (NO TELEMETRY / NO ROBOT TEXT)
        story_opts = STORY_TEMPLATES.get(detected_archetype, STORY_TEMPLATES["general"])
        context_story = random.choice(story_opts).format(creator=creator)

        # If clean narrative summary is available and free of telemetry, prefer it
        if raw_summary and len(raw_summary.strip()) > 10 and not any(w in raw_summary.lower() for w in ("chat surged", "baseline", "messages in")):
            context_story = raw_summary.strip()

        description = cls._format_structured_description(
            hook=hook,
            summary=context_story,
            creator=creator,
            platform=platform,
            cta=cta,
            tags=tags,
        )

        return ViralMetadata(
            title=title[:100],
            suggested_hook=hook,
            description=description,
            tags=tags,
            category=category,
            call_to_action=cta,
        )

    @classmethod
    def _detect_archetype(cls, dialogue: str, category: str, spike_ratio: float) -> str:
        """Analyze dialogue keywords and spike metrics to categorize emotional hook."""
        if spike_ratio >= 4.0:
            return "chat"

        d_lower = dialogue.lower()
        cat_lower = category.lower()

        if any(w in d_lower for w in ("clutch", "headshot", "kill", "win", "ace", "sniper", "gg")) or cat_lower in ("clutch", "skillful", "impressive"):
            return "clutch"
        if any(w in d_lower for w in ("no way", "what the", "how", "wait", "holy", "freeze", "what did")) or cat_lower in ("unexpected", "dramatic"):
            return "shock"
        if any(w in d_lower for w in ("haha", "lol", "lmao", "funny", "bro", "dead", "joke", "crying")) or cat_lower in ("funny", "fail", "chaotic"):
            return "funny"

        return "general"

    @classmethod
    def _extract_punchy_quote(cls, dialogue: str) -> str | None:
        """Extract short dramatic quote from dialogue if suitable for title."""
        if not dialogue:
            return None
        parts = re.split(r"[.?!]", dialogue)
        for part in parts:
            clean = part.strip(' "\'()[]')
            word_count = len(clean.split())
            if 2 <= word_count <= 6 and len(clean) <= 35:
                if not any(f == clean.lower() for f in ("thank you", "hello", "hi guys", "one sec", "okay")):
                    if not any(bad in clean.lower() for bad in ("chat surged", "baseline", "messages in", "dialogue")):
                        return clean
        return None

    @classmethod
    def _build_tags(cls, creator: str, platform: str, category: str, dialogue: str) -> list[str]:
        """Generate high-performing YouTube Shorts & SEO tag list."""
        base_tags = [
            "Shorts",
            "YouTubeShorts",
            "viral",
            "trending",
            "gaming",
            "gaming clips",
            "funny moments",
            "stream highlights",
            "twitch clips",
            "kick clips",
            "live stream",
            "reaction",
            "instant regret",
            "clutch",
            "fails",
        ]

        creator_tags = []
        if creator and creator.lower() not in ("streamer", "unknown"):
            c = creator.strip()
            creator_tags = [
                c,
                f"{c} clips",
                f"{c} highlights",
                f"{c} stream",
                f"{c} funny",
                f"{c} shorts",
            ]

        platform_tags = [platform, f"{platform} live", f"{platform} stream"]

        all_tags = []
        seen = set()
        for t in creator_tags + platform_tags + base_tags:
            t_norm = t.lower().strip()
            if t_norm and t_norm not in seen:
                seen.add(t_norm)
                all_tags.append(t)

        return all_tags[:20]

    @classmethod
    def _format_structured_description(
        cls,
        hook: str,
        summary: str,
        creator: str,
        platform: str,
        cta: str,
        tags: list[str],
    ) -> str:
        """Standard high-converting YouTube Shorts description layout with natural storytelling."""
        creator_tag = f"@{creator.lstrip('@')}" if creator else "@Streamer"
        hashtag_cloud = " ".join([f"#{t.replace(' ', '')}" for t in tags[:10] if not t.startswith("#")])

        lines = [
            f"{hook} 🔥💀",
            "",
            summary.strip(),
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📺 Streamer: {creator_tag}",
            "🌟 Daily Viral Stream Highlights & Clutch Reactions",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"💬 {cta}",
            "👉 LIKE & SUBSCRIBE for daily viral gaming shorts! 🔔",
            "",
            f"#Shorts {hashtag_cloud}",
        ]
        return "\n".join(lines).strip()
