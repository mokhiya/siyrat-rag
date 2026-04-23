# evaluation/judge.py
# LLM-as-judge: score benchmark answers on faithfulness, accuracy, and completeness.
# Run after benchmark.py has produced results/benchmark_results.json.
import sys
import os
import json
import time
import re

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "results", "benchmark_results.json")
SCORES_PATH  = os.path.join(BASE_DIR, "results", "benchmark_scores.json")

JUDGE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

JUDGE_PROMPT = """Sen obyektiv baholovchisan. Quyidagi RAG tizimi javobini 3 mezon bo'yicha baho.

SAVOL: {question}
TO'G'RI JAVOB: {expected}

BAHOLANAYOTGAN JAVOB:
{answer}

Har mezon uchun 1-5 ball ber va qisqa izoh yoz. Javobingni faqat JSON formatida ber:

{{
  "faithfulness": {{
    "score": <1-5>,
    "reason": "<manbaga sodiqmi? o'zidan qo'shganmi?>"
  }},
  "accuracy": {{
    "score": <1-5>,
    "reason": "<faktlar to'g'rimi? to'g'ri javobga mos keladi?>"
  }},
  "completeness": {{
    "score": <1-5>,
    "reason": "<savolga to'liq javob berdimi?>"
  }}
}}

Balllar:
1 = Juda yomon  2 = Yomon  3 = O'rtacha  4 = Yaxshi  5 = A'lo"""


def judge_answer(groq_client, question, expected, answer):
    """Score a single answer using the LLM judge. Returns a dict with three criteria."""
    if answer.startswith("ERROR:"):
        return {
            "faithfulness": {"score": 0, "reason": "API error"},
            "accuracy":     {"score": 0, "reason": "API error"},
            "completeness": {"score": 0, "reason": "API error"},
        }

    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    try:
        response = groq_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]+\}', text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"      ⚠️ Judge error: {e}")

    return {
        "faithfulness": {"score": 0, "reason": "Parse error"},
        "accuracy":     {"score": 0, "reason": "Parse error"},
        "completeness": {"score": 0, "reason": "Parse error"},
    }


def run_judge():
    print("Loading benchmark results...")
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    models      = data["models"]
    results     = data["results"]

    scores = []
    total  = len(results) * len(models)
    done   = 0

    for row in results:
        score_row = {
            "id":         row["id"],
            "question":   row["question"],
            "difficulty": row["difficulty"],
            "scores":     {},
        }

        for model in models:
            done += 1
            print(f"\n[{done}/{total}] {model.split('/')[0]} | {row['question'][:50]}")
            answer   = row["answers"].get(model, "No answer")
            judgment = judge_answer(groq_client, row["question"], row["expected"], answer)

            score_row["scores"][model] = {
                "answer":    answer,
                "judgment":  judgment,
                "avg_score": round(
                    (judgment.get("faithfulness", {}).get("score", 0) +
                     judgment.get("accuracy",     {}).get("score", 0) +
                     judgment.get("completeness", {}).get("score", 0)) / 3, 2
                ),
            }
            f_s = judgment.get("faithfulness", {}).get("score", "?")
            a_s = judgment.get("accuracy",     {}).get("score", "?")
            c_s = judgment.get("completeness", {}).get("score", "?")
            print(f"   F:{f_s}  A:{a_s}  C:{c_s}  avg:{score_row['scores'][model]['avg_score']}")
            time.sleep(2)

        scores.append(score_row)

    # Aggregate summary statistics per model
    summary = {}
    for model in models:
        model_scores = [
            row["scores"][model]["avg_score"]
            for row in scores
            if model in row["scores"]
        ]
        by_difficulty = {}
        for diff in ["easy", "medium", "hard"]:
            diff_scores = [
                row["scores"][model]["avg_score"]
                for row in scores
                if row["difficulty"] == diff and model in row["scores"]
            ]
            by_difficulty[diff] = (
                round(sum(diff_scores) / len(diff_scores), 2) if diff_scores else 0
            )
        summary[model] = {
            "overall_avg":  round(sum(model_scores) / len(model_scores), 2) if model_scores else 0,
            "by_difficulty": by_difficulty,
        }

    output = {
        "date":    data["date"],
        "models":  models,
        "summary": summary,
        "details": scores,
    }

    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("FINAL RESULTS:")
    for model, s in summary.items():
        name = model.split('/')[0].split('-')[0].upper()
        diff = s["by_difficulty"]
        print(f"  {name}: {s['overall_avg']}/5 | "
              f"easy:{diff['easy']}  medium:{diff['medium']}  hard:{diff['hard']}")
    print(f"{'='*50}")
    print(f"\n✅ Scores saved: {SCORES_PATH}")


if __name__ == "__main__":
    run_judge()
