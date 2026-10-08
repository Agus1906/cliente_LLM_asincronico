"""Demostración real: pregunta breve en modo completo y streaming."""
import argparse
import asyncio
import logging
from contextlib import aclosing
from pathlib import Path

from dotenv import load_dotenv

from manager import AsyncLLMManager, ConfigurationError
from schemas import ChatMessage


async def demonstrate(provider: str | None) -> bool:
    try:
        async with AsyncLLMManager.from_env(provider) as manager:
            messages = [
                ChatMessage(role="system", content="Respondé en español, en un solo párrafo de como máximo 60 palabras."),
                ChatMessage(role="user", content="¿Qué es la entropía?"),
            ]
            print(f"\nProveedor: {manager.client.provider}\nModo normal:")
            response = await manager.generate(messages)
            print(response.content if response.ok else response.error.model_dump_json())
            if response.ok:
                print(f"Fin: {response.finish_reason}")
            print("\nStreaming: ", end="", flush=True)
            success = response.ok
            async with aclosing(manager.stream(messages)) as events:
                async for event in events:
                    if event.type == "text":
                        print(event.text, end="", flush=True)
                    elif event.type == "error":
                        print(f"\nError: {event.error.model_dump_json()}")
                        success = False
                    else:
                        print(f"\nFin: {event.finish_reason}")
            return success
    except ConfigurationError as exc:
        print(f"Configuración: {exc}")
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "anthropic", "both"])
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    providers = ["openai", "anthropic"] if args.provider == "both" else [args.provider]
    results = [await demonstrate(provider) for provider in providers]
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nEjecución cancelada.")
        raise SystemExit(130)
