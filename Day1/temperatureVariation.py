import os
import csv
from groq import Groq

os.environ["GROQ_API_KEY"] = ""

client = Groq()

prompts = [
    "Explain photosynthesis in one sentence.",
    "Write the opening line of a sci-fi novel.",
    "What shoud I do when driving a car downhill?",
    "Which city is the capital of Islamabad? Briefly mention its history.",
    "Translate 'Where is the library?' into German.",
    "Summarize: The quick brown fox jumps over the lazy dog. The dog was not amused.",
    "Is pineapple on pizza acceptable? Why or why not?",
    "Give me a 3-bullet list of benefits of sleep.",
    "Tell me lines losses in electric grid, in  abrief manner.",
    "Name a color, then name an animal that color, then make a sentence.",
]

temps = [0, 0.7, 1.0]
MODEL = "openai/gpt-oss-120b"

results = []

for p in prompts:
    print(f"\nPROMPT: {p}")
    for t in temps:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": p}],
            temperature=t,
            max_tokens=400
        )
        out = resp.choices[0].message.content
        results.append({"prompt": p, "temperature": t, "output": out})
        print(f"  [Temp={t}] Done")

# Save to CSV
csv_filename = "temperature_results.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "temperature", "output"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {len(results)} rows to {csv_filename}")
