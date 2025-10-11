"""Reproduce textos como dictados desde la línea de comandos.

El programa permite practicar la transcripción manual de textos: separa las
palabras y signos de puntuación para leerlos uno a uno con un intervalo de
tiempo configurable. Puede proporcionar el contenido directamente con la opción
``--text`` o leerlo desde un archivo mediante ``--file``.

Ejemplos de uso
---------------

Dictar un párrafo con una pausa de 1.8 segundos entre cada palabra::

    python dictado.py --text "Hola, ¿cómo estás?" --delay 1.8

Leer un archivo y aumentar la pausa cuando termina una oración::

    python dictado.py --file cuento.txt --sentence-delay 2.5

Las pausas añadidas sobre signos de puntuación ayudan a interiorizar la
cadencia de la lectura en voz alta. Los signos más habituales se convierten a
su denominación en español (por ejemplo, `,` se dicta como "coma").
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List

import re


# Signos de puntuación comunes y cómo deben pronunciarse en el dictado.
PUNCTUATION_MAP = {
    ".": "punto",
    ",": "coma",
    ";": "punto y coma",
    ":": "dos puntos",
    "?": "signo de interrogación",
    "¿": "signo de interrogación abierto",
    "!": "signo de exclamación",
    "¡": "signo de exclamación abierto",
    "\"": "comillas",
    "'": "comilla simple",
    "«": "comillas angulares abiertas",
    "»": "comillas angulares cerradas",
    "(": "paréntesis abre",
    ")": "paréntesis cierra",
    "[": "corchete abre",
    "]": "corchete cierra",
    "{": "llave abre",
    "}": "llave cierra",
    "-": "guion",
    "—": "raya",
    "…": "puntos suspensivos",
    "/": "barra",
}

# Signos que suelen marcar el final de una oración. Añadimos una pausa extra
# para permitir que la persona que escribe alcance el ritmo del dictado.
SENTENCE_ENDINGS = {".", "?", "!", "¿", "¡", "…"}

# Separar palabras y signos de puntuación sin perder estos últimos.
TOKEN_PATTERN = re.compile(r"…|\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Devuelve la secuencia de palabras y signos de puntuación."""

    return TOKEN_PATTERN.findall(text)


def describe_token(token: str) -> str:
    """Convierte el token en la palabra que debe pronunciarse."""

    if token in PUNCTUATION_MAP:
        return PUNCTUATION_MAP[token]

    if token.isspace():
        return "espacio"

    # Los tokens no incluidos en el mapa se dictan literalmente.
    return token


def iter_tokens_with_pauses(
    tokens: Iterable[str],
    base_delay: float,
    punctuation_extra: float,
    sentence_delay: float,
) -> Iterator[float]:
    """Itera sobre los tokens y produce el tiempo a esperar tras cada uno."""

    for token in tokens:
        description = describe_token(token)
        print(description, flush=True)

        delay = base_delay
        if token in PUNCTUATION_MAP:
            delay += punctuation_extra
        if token in SENTENCE_ENDINGS:
            delay += sentence_delay

        yield delay


def load_text(args: argparse.Namespace) -> str:
    """Obtiene el texto ya sea desde ``--text`` o desde ``--file``."""

    if args.text is not None:
        return args.text

    if args.file is None:
        raise ValueError("Debe proporcionar --text o --file.")

    try:
        return Path(args.file).read_text(encoding=args.encoding)
    except OSError as exc:  # pragma: no cover - manejo defensivo de IO.
        raise SystemExit(f"No se pudo leer '{args.file}': {exc}") from exc


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dictado interactivo con control del tiempo entre palabras.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--text",
        help="Texto a dictar directamente desde la línea de comandos.",
    )
    source.add_argument(
        "--file",
        help="Ruta a un archivo de texto que se dictará.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Codificación a utilizar al leer un archivo (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Segundos a esperar entre cada palabra (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--punctuation-delay",
        type=float,
        default=0.5,
        help=(
            "Tiempo adicional en segundos cuando se dicta un signo de puntuación "
            "(por defecto: %(default)s)."
        ),
    )
    parser.add_argument(
        "--sentence-delay",
        type=float,
        default=1.0,
        help=(
            "Pausa extra tras los signos que cierran una oración, en segundos "
            "(por defecto: %(default)s)."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Cantidad de veces que se repetirá el dictado completo.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Muestra los tokens que se dictarán antes de iniciar.",
    )
    return parser.parse_args(argv)


def run_dictation(args: argparse.Namespace) -> None:
    text = load_text(args)
    tokens = tokenize(text)

    if args.preview:
        print("Vista previa del dictado:")
        print(" | ".join(describe_token(token) for token in tokens))
        print()

    for repetition in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"— Dictado {repetition} de {args.repeat} —")

        for delay in iter_tokens_with_pauses(
            tokens,
            base_delay=max(args.delay, 0.0),
            punctuation_extra=max(args.punctuation_delay, 0.0),
            sentence_delay=max(args.sentence_delay, 0.0),
        ):
            time.sleep(delay)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    run_dictation(args)


if __name__ == "__main__":
    main()
