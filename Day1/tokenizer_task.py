from transformers import AutoTokenizer
import tiktoken
import csv

# Two different tokenizers — both offline, zero setup
gpt4 = tiktoken.get_encoding("cl100k_base")          # GPT-4 tokenizer
gpt2 = AutoTokenizer.from_pretrained("gpt2")         # GPT-2 tokenizer (different vocab!)

strings = [
    'print("hello")',
    'def factorial(n):\n    if n == 0:\n        return 1',
    'const x = {a: 1, b: [2,3,4]};',
    '{"key": "value", "num": 42, "nested": {"a": 1}}',
    'SELECT * FROM users WHERE id = 1;',
    '<html><body><h1>Hello</h1></body></html>',
    'x = 5 + 3 * (2 - 8) / 4',
    'README.md',
    'میرا نام John ہے',
    'Ich liebe KI und machine learning',
    'Dies ist ein Test mit 123 Zahlen',
    'مرحبا بالعالم',
    'नमस्ते दुनिया',
    '你好世界',
    '👨‍👩‍👧‍👦',
    '🚀🔥💯',
    'Hello 👋 world 🌍!!!',
    '🇺🇸🇬🇧🇯🇵',
    'https://example.com/very/long/path?a=1&b=2&c=3',
    'https://sub.domain.co.uk/another/very/long/path/with/many/segments',
    'https://t.co/veryshort',
    'aaaaaaaaaaaaaaaaaaaa',
    '1234567890',
    '1,234,567.89',
    'COVID-19',
    'supercalifragilisticexpialidocious',
    '   \n\n\t   ',
    '¯\\_(ツ)_/¯',
    '!!!???...,,,',
    'The quick brown fox jumps over the lazy dog.',
]

results = []
for s in strings:
    gpt4_count = len(gpt4.encode(s))
    gpt2_count = len(gpt2.encode(s))
    results.append({
        "string": s,
        "gpt4_tokens": gpt4_count,
        "gpt2_tokens": gpt2_count,
        "difference": gpt2_count - gpt4_count,
        "ratio": round(gpt2_count / gpt4_count, 2) if gpt4_count else None
    })
    print(f"{s[:40]:<40} | GPT-4: {gpt4_count:>3} | GPT-2: {gpt2_count:>3}")

with open("tokenizer_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["string", "gpt4_tokens", "gpt2_tokens", "difference", "ratio"])
    writer.writeheader()
    writer.writerows(results)

print("\n✅ Saved to tokenizer_comparison.csv")