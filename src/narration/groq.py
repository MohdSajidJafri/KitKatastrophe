from typing import Optional
import config
from src.logger import logger

SYSTEM_PROMPT_CURATOR = """You are an expert social media meme editor who writes viral, punchy captions for funny & cute cat video shorts (like Twitter/X / Instagram cat reels).

Task:
Convert the provided Reddit post title into a snappy, engaging 4 to 12 word caption highlighting the cat's hilarious behavior, cute antics, or playful chaos.

Guidelines:
1. WORD COUNT: Exactly 4 to 12 words. Neither too short nor too long.
2. RELEVANCE: Directly resemble and reflect the funny/cute cat behavior in the video based on the post context.
3. VAGUE TITLES: If the Reddit title is lazy, vague, or generic (e.g. 'An interesting title', 'Title', 'Cat', 'Wait for it', 'lol', 'Seen today', 'Wait till the end', 'Meow'), generate a relatable, punchy title suited for funny and cute cats (e.g. 'Zero thoughts behind those eyes 🐱', 'The cat software crashed again 💀', 'Single brain cell operating at 1% 😹', 'He woke up and chose chaos 😼', 'Peak cat behavior 😂').
4. STYLE: Modern cat lover & internet meme humor (e.g. 'Catculations were slightly off 😂', 'Orange cat activities 😭', 'Standard orange cat behavior 🐱', 'Not a single thought in that head 😂').
5. EMOJIS: Include 1-2 relevant cute/funny emojis (🐱, 😹, 😂, 😭, 💀, 🐾, etc.) at the end.
6. FORMAT: Return ONLY the caption on a single line. Do not include quotes, prefixes, or explanations."""


def generate_curator_caption(title: str, subreddit: str = "") -> str:
    """
    Use Groq to convert raw Reddit titles into 4-12 word viral, snappy captions.
    If no API key is provided or the call fails, cleanly returns the title.
    """
    api_key = config.GROQ_API_KEY
    if not api_key:
        return _fallback_curator_caption(title)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        return _fallback_curator_caption(title)

    models_to_try = [
        config.LLM_MODEL,
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]
    seen = set()
    deduped_models = []
    for m in models_to_try:
        if m and m not in seen:
            seen.add(m)
            deduped_models.append(m)

    prompt = f"Subreddit: r/{subreddit}\nReddit Title: {title}\nCaption:"
    for m in deduped_models:
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CURATOR},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=60,
                timeout=15
            )
            raw_caption = completion.choices[0].message.content
            if raw_caption:
                cleaned = raw_caption.strip().strip('"\'')
                for prefix in ["Caption:", "Title:", "CAPTION:", "TITLE:"]:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip().strip('"\'')
                
                words = cleaned.split()
                if 2 <= len(words) <= 15:
                    logger.info(f"✔ Groq generated viral curator caption: '{cleaned}' ({len(words)} words)")
                    return cleaned
                elif len(words) > 15:
                    cleaned = " ".join(words[:12])
                    return cleaned
        except Exception:
            continue

    return _fallback_curator_caption(title)


def _fallback_curator_caption(title: str) -> str:
    """Safe local fallback for caption generation."""
    clean = title.strip()
    words = clean.split()
    vague_patterns = [
        "an interesting title", "title", "cat", "cats", "wait for it",
        "wait till the end", "lol", "interesting", "seen today", "meow"
    ]
    if clean.lower().strip(".,!?:") in vague_patterns or len(words) <= 1:
        return "Zero thoughts behind those eyes 🐱"
    
    if len(words) > 12:
        return " ".join(words[:12]) + " 🐱"
    return clean
