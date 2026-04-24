import asyncio
import time
from decimal import Decimal

import httpx

_QUESTIONS = [
    "What is the objective of the assignment?",
    "What file types can be ingested?",
    "What is the time expectation?",
    "What are the bonus requirements?",
    "What should the API endpoints be?",
]


async def _ask(client: httpx.AsyncClient, question: str) -> dict:
    started = time.perf_counter()
    response = await client.post(
        "http://localhost:8000/api/v1/ask",
        json={"question": question},
        timeout=60.0,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": response.status_code,
        "latency_s": elapsed,
        "body": response.json(),
        "question": question,
    }


async def main() -> None:
    async with httpx.AsyncClient() as client:
        wall_started = time.perf_counter()
        results = await asyncio.gather(*(_ask(client, q) for q in _QUESTIONS))
        wall = time.perf_counter() - wall_started

    print(f"\n=== 5 concurrent /ask — wall-clock {wall:.2f}s ===\n")
    total_cost = Decimal("0")
    for index, result in enumerate(results):
        body = result["body"]
        cost = Decimal(body["usage"]["estimated_cost_usd"])
        total_cost += cost
        answer = body["answer"][:90].replace("\n", " ")
        print(
            f"#{index}  status={result['status']}  latency={result['latency_s']:.2f}s  "
            f"cache_hit={body['usage']['cache_hit']}  grounded={body['is_grounded']}  "
            f"cost=${cost}"
        )
        print(f"      -> {answer}")
    print(f"\ntotal_cost=${total_cost}")
    print(
        f"serial_estimate={sum(r['latency_s'] for r in results):.2f}s  "
        f"parallel_speedup={sum(r['latency_s'] for r in results) / wall:.2f}x"
    )


if __name__ == "__main__":
    asyncio.run(main())
