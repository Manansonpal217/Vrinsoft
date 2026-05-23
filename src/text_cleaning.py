"""Noisy social-media text cleaning and tokenisation.

Twitter customer-support text is messy: @mentions, t.co links, HTML entities
(`&amp;`, `&gt;`), emoji, agent sign-offs (`^RR`, `/AY`, `-KC`), elongated
words (`soooo`) and so on.  This module normalises all of that into a clean
token stream the sequential model can learn from.

Placeholder tokens used (kept as single tokens by the tokenizer):
    <user>   - an @mention (brand handle or anonymised customer id)
    <url>    - any hyperlink
    <emoji>  - one or more emoji characters
    <num>    - a standalone number / version string component
"""
import html
import re

# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+|\S+\.co/\S+)")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
# Agent sign-offs at the end of a support reply, e.g. " ^RR", " /AY", " -KC".
_SIGNOFF_RE = re.compile(r"\s[\^/*\-][A-Za-z]{1,4}\s*$")
# Common emoji / pictograph unicode ranges.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # symbols, pictographs, supplemental
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002190-\U000021FF"   # arrows
    "\U00002B00-\U00002BFF"   # misc symbols & arrows
    "\U0000FE0F\U0000200D\U000020E3"  # variation selector / ZWJ / keycap
    "]+",
    flags=re.UNICODE,
)
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
# word | contraction | number | single non-space symbol
_TOKEN_RE = re.compile(r"<[a-z]+>|[a-z]+(?:'[a-z]+)?|[0-9]+|[^\sa-z0-9]")
_DIGITS_RE = re.compile(r"^\d+$")


def clean_text(text: str) -> str:
    """Normalise a raw tweet into lower-cased, placeholder-substituted text."""
    if text is None:
        return ""
    text = str(text)

    # 1. strip the agent sign-off before anything else (it relies on caps).
    text = _SIGNOFF_RE.sub("", text)
    # 2. decode HTML entities (&amp; -> &, &gt; -> > ...).
    text = html.unescape(text)
    # 3. swap hyperlinks and @mentions for placeholders.
    text = _URL_RE.sub(" <url> ", text)
    text = _MENTION_RE.sub(" <user> ", text)
    # 4. keep the word part of a hashtag (#refund -> refund).
    text = _HASHTAG_RE.sub(r" \1 ", text)
    # 5. collapse emoji runs into a single placeholder.
    text = _EMOJI_RE.sub(" <emoji> ", text)
    # 6. lower-case and squash elongated runs (soooo -> soo, !!!! -> !!).
    text = text.lower()
    text = _ELONGATION_RE.sub(r"\1\1", text)
    # 7. tidy whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Tokenise already-cleaned text into a list of tokens.

    Pure-digit tokens are mapped to the <num> placeholder so the model is not
    overwhelmed by phone numbers, order ids and version strings.
    """
    tokens = _TOKEN_RE.findall(text)
    return ["<num>" if _DIGITS_RE.match(t) else t for t in tokens]


def clean_and_tokenize(text: str) -> list:
    """Convenience: clean a raw tweet and return its token list."""
    return tokenize(clean_text(text))


if __name__ == "__main__":  # quick manual smoke test
    samples = [
        "@AppleSupport causing the reply to be disregarded 😡😡😡",
        "@115712 Please send us a Private Message so that we can assist. -KC",
        "I still haven't heard &amp; the number is a dead end. Call me? https://t.co/abc",
        "hi #apple, iOS is sooooo slow on #iphone6!!!!",
    ]
    for s in samples:
        print(f"RAW : {s}")
        print(f"CLEAN: {clean_text(s)}")
        print(f"TOKS: {clean_and_tokenize(s)}\n")
