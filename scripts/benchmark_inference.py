"""推理性能压测：TTFT、P50/P95、吞吐、RPS、显存。

用法：
  # 压测本地心语 API（SSE）
  python scripts/benchmark_inference.py --mode app --url http://localhost:8010 --label offline-api

  # 压测 vLLM OpenAI 接口
  python scripts/benchmark_inference.py --mode openai --url http://localhost:8001/v1 \
      --model heart-say-qwen --label qwen7b-awq
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROMPTS = [
    "孩子最近总说不想上学，我该怎么开口？",
    "我一考试就睡不着，怕考砸。",
    "孩子把自己关在房间，不跟我说话。",
    "我总控制不住吼孩子，事后很自责。",
    "孩子被同学起外号，回家一直哭。",
    "我女儿最近很在意身材，不太吃饭。",
    "孩子沉迷手机，我们天天吵架。",
    "离婚后孩子变得沉默，我该怎么陪他？",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return round(ordered[index], 1)


def gpu_memory() -> dict | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        used, total = [int(value.strip()) for value in output.splitlines()[0].split(",")]
        return {"used_mb": used, "total_mb": total, "used_gb": round(used / 1024, 2)}
    except Exception:
        return None


async def call_openai(client: httpx.AsyncClient, args, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first = None
    text = ""
    usage = {}
    async with client.stream(
        "POST",
        f"{args.url.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=args.timeout,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}).get("content") or ""
            if delta and first is None:
                first = time.perf_counter()
            text += delta
    total = time.perf_counter()
    tokens = int(usage.get("completion_tokens") or max(1, len(text) // 2))
    return {
        "ttft_ms": round(((first or total) - started) * 1000, 1),
        "total_ms": round((total - started) * 1000, 1),
        "tokens": tokens,
        "text": text,
    }


async def call_app(client: httpx.AsyncClient, args, prompt: str) -> dict:
    payload = {
        "message": prompt,
        "session_id": f"bench{uuid.uuid4().hex[:8]}",
        "stream": True,
        "use_rag": True,
    }
    started = time.perf_counter()
    first = None
    text = ""
    event = ""
    async with client.stream(
        "POST", f"{args.url.rstrip('/')}/chat/stream", json=payload, timeout=args.timeout
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            try:
                payload_data = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event == "delta":
                chunk = payload_data.get("text", "")
                if chunk and first is None:
                    first = time.perf_counter()
                text += chunk
            elif event == "error":
                raise RuntimeError(payload_data.get("message", "SSE error"))
    total = time.perf_counter()
    tokens = max(1, len(text) // 2)
    return {
        "ttft_ms": round(((first or total) - started) * 1000, 1),
        "total_ms": round((total - started) * 1000, 1),
        "tokens": tokens,
        "text": text,
    }


async def run(args) -> dict:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        async def one(prompt: str):
            async with semaphore:
                if args.mode == "openai":
                    return await call_openai(client, args, prompt)
                return await call_app(client, args, prompt)

        results = await asyncio.gather(*[one(prompt) for prompt in prompts], return_exceptions=True)
    wall = time.perf_counter() - started

    errors = [str(item) for item in results if isinstance(item, Exception)]
    successes = [item for item in results if not isinstance(item, Exception)]
    if not successes:
        raise RuntimeError(f"全部请求失败：{errors[:2]}")

    ttfts = [item["ttft_ms"] for item in successes]
    totals = [item["total_ms"] for item in successes]
    tokens = sum(item["tokens"] for item in successes)
    report = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "url": args.url,
        "model": args.model if args.mode == "openai" else "heart-say-api",
        "requests": args.requests,
        "concurrency": args.concurrency,
        "success": len(successes),
        "errors": len(errors),
        "ttft_ms": {"p50": percentile(ttfts, 0.5), "p95": percentile(ttfts, 0.95)},
        "latency_ms": {"p50": percentile(totals, 0.5), "p95": percentile(totals, 0.95)},
        "throughput_tokens_per_sec": round(tokens / wall, 1),
        "rps": round(len(successes) / wall, 2),
        "wall_sec": round(wall, 2),
        "avg_output_tokens": round(tokens / len(successes), 1),
        "gpu_memory": gpu_memory(),
    }
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"perf_{args.label}.json"
    md_path = out_dir / f"perf_{args.label}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    vram = report["gpu_memory"]
    md = "\n".join(
        [
            f"# 推理性能报告 · {args.label}",
            "",
            f"- 时间：{report['generated_at']}",
            f"- 模式：`{args.mode}`；地址：{args.url}",
            f"- 请求：{args.requests}，并发：{args.concurrency}，成功：{report['success']}，失败：{report['errors']}",
            f"- TTFT P50/P95：**{report['ttft_ms']['p50']} / {report['ttft_ms']['p95']} ms**",
            f"- 总延迟 P50/P95：{report['latency_ms']['p50']} / {report['latency_ms']['p95']} ms",
            f"- 吞吐：{report['throughput_tokens_per_sec']} tokens/s；RPS：{report['rps']}",
            f"- 平均输出：{report['avg_output_tokens']} tokens",
            f"- 显存：{json.dumps(vram, ensure_ascii=False) if vram else '未检测到 NVIDIA GPU'}",
            "",
            "> 延迟包含网络与检索；与 vLLM 直连数字对比时需说明口径。",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["app", "openai"], default="app")
    parser.add_argument("--url", default="http://localhost:8010")
    parser.add_argument("--model", default="heart-say-qwen")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--label", default="offline-api")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
