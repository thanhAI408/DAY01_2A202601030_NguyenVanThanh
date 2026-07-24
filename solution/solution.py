import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
}
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


def _client():
    from openai import OpenAI
    kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_openai(prompt: str, model: str = OPENAI_MODEL, temperature: float = 0.7,
                top_p: float = 0.9, max_tokens: int = 256) -> tuple[str, float]:
    client = _client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = max(time.perf_counter() - start, 1e-9)
    return response.choices[0].message.content or "", latency


def call_openai_mini(prompt: str, temperature: float = 0.7, top_p: float = 0.9,
                     max_tokens: int = 256) -> tuple[str, float]:
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens)


def compare_models(prompt: str) -> dict:
    gpt4o_answer, gpt4o_time = call_openai(prompt)
    mini_answer, mini_time = call_openai_mini(prompt)
    pricing = PRICING_PER_1K_TOKENS.get(
        OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"]
    )
    estimated_output_tokens = len(gpt4o_answer.split()) / 0.75
    gpt4o_cost = estimated_output_tokens / 1000 * pricing["output"]
    return {
        "gpt4o_answer": gpt4o_answer,
        "mini_answer": mini_answer,
        "gpt4o_time": gpt4o_time,
        "mini_time": mini_time,
        "gpt4o_cost": gpt4o_cost,
    }


def chat_with_system_prompt(system_prompt: str, user_prompt: str,
                            model: str = OPENAI_MODEL, temperature: float = 0.7,
                            max_tokens: int = 256) -> tuple[str, float]:
    client = _client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = max(time.perf_counter() - start, 1e-9)
    return response.choices[0].message.content or "", latency


def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    prompt_tokens = count_tokens(prompt, model)
    completion_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(
        model, PRICING_PER_1K_TOKENS["gpt-4o"]
    )
    prompt_cost = prompt_tokens / 1000 * pricing["input"]
    completion_cost = completion_tokens / 1000 * pricing["output"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": prompt_cost + completion_cost,
    }


def streaming_chatbot() -> None:
    client = _client()
    history = []
    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit", "bye"):
            break
        messages = history + [{"role": "user", "content": user_msg}]
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True
        )
        reply = ""
        print("Trợ lý: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ])
        history = history[-8:]


def retry_with_backoff(fn: Callable, max_retries: int = 3,
                       base_delay: float = 0.1) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


def run_assistant(persona: str, get_input: Callable[[], str] = None,
                  max_turns: int = None) -> dict:
    if get_input is None:
        get_input = input
    client = _client()
    history = []
    turns = 0
    tokens_used = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and turns >= max_turns:
            break
        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit", "bye"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()

        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ])
        history = history[-8:]
        turns += 1
        tokens_used += count_tokens(user_msg, OPENAI_MODEL)
        tokens_used += count_tokens(reply, OPENAI_MODEL)
        total_cost += estimate_cost(user_msg, reply, OPENAI_MODEL)["total_cost"]

    return {
        "turns": turns,
        "tokens_used": tokens_used,
        "total_cost": total_cost,
        "history": history,
    }


def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


def format_comparison_table(results: list[dict]) -> str:
    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    rows = [headers]
    for r in results:
        rows.append([
            str(r.get("prompt", ""))[:40],
            str(r.get("gpt4o_answer", ""))[:40],
            str(r.get("mini_answer", ""))[:40],
            f"{float(r.get('gpt4o_time', 0)):.3f}s",
            f"{float(r.get('mini_time', 0)):.3f}s",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)
