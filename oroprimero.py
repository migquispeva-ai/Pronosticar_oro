"""Herramienta de pronóstico para la onza de oro.

Este módulo descarga precios históricos del oro, recopila noticias
financieras recientes para estimar el sentimiento económico y utiliza un
modelo econométrico sencillo para predecir los rendimientos futuros.

La aplicación se diseñó como una referencia reproducible: combina
indicadores cuantitativos (rendimientos rezagados) con un factor
cualitativo (sentimiento de noticias). El modelo puede refinarse con
datos adicionales, diferentes fuentes de noticias o estrategias de
modelado más avanzadas.

Ejemplo de uso desde la línea de comandos::

    python oroprimero.py --horizon 5 --export pronosticos.csv

Si define la variable de entorno ``NEWS_API_KEY`` con una clave válida
de https://newsapi.org, el script incorporará automáticamente noticias
financieras recientes en español. En caso contrario, el factor de
sentimiento se establece en cero, permitiendo que el modelo funcione de
forma puramente econométrica.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd
import requests
import statsmodels.api as sm

try:  # La librería puede no estar instalada; degradamos con elegancia.
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk import download as nltk_download

    _VADER_AVAILABLE = True
except Exception:  # pragma: no cover - comportamiento defensivo.
    SentimentIntensityAnalyzer = None  # type: ignore
    nltk_download = None  # type: ignore
    _VADER_AVAILABLE = False

try:  # ``yfinance`` facilita el acceso a precios históricos.
    import yfinance as yf

    _YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - manejo defensivo.
    yf = None  # type: ignore
    _YFINANCE_AVAILABLE = False


DEFAULT_TICKER = "GC=F"  # Futuro de oro COMEX como proxy del oro spot.
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"


@dataclass
class PredictionResult:
    """Representa la salida del modelo de pronóstico."""

    forecast: pd.DataFrame
    model_summary: str
    sentiment_used: bool


def configure_logging(verbose: bool) -> None:
    """Configura la salida de logging."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def ensure_vader() -> Optional[SentimentIntensityAnalyzer]:
    """Devuelve un analizador VADER si está disponible."""

    if not _VADER_AVAILABLE:
        logging.warning(
            "VADER (nltk) no está disponible. El sentimiento se tratará como neutral."
        )
        return None

    try:
        nltk_download("vader_lexicon", quiet=True)
    except Exception as exc:  # pragma: no cover - descarga puede fallar.
        logging.warning("No fue posible descargar el léxico VADER: %s", exc)

    try:
        return SentimentIntensityAnalyzer()
    except Exception as exc:  # pragma: no cover - inicialización defensiva.
        logging.warning("No se pudo inicializar VADER: %s", exc)
        return None


def fetch_gold_prices(
    ticker: str = DEFAULT_TICKER,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
) -> pd.DataFrame:
    """Descarga precios históricos del oro utilizando ``yfinance``."""

    if not _YFINANCE_AVAILABLE:
        raise RuntimeError(
            "La librería yfinance no está disponible. Instálela con 'pip install yfinance'."
        )

    if end is None:
        end = dt.date.today()
    if start is None:
        start = end - dt.timedelta(days=365 * 5)

    logging.info(
        "Descargando precios de %s entre %s y %s", ticker, start.isoformat(), end.isoformat()
    )
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("No se obtuvieron precios históricos. Verifique el ticker o la conexión.")

    data = data.rename(columns=str.lower)
    data = data[["close"]]
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    logging.debug("Se descargaron %d observaciones de precios", len(data))
    return data


def fetch_financial_news(
    api_key: Optional[str],
    from_date: dt.date,
    to_date: dt.date,
    language: str = "es",
    query: str = "oro precio economia OR inflation OR dolar",
) -> List[dict]:
    """Consulta titulares financieros relevantes utilizando NewsAPI."""

    if not api_key:
        logging.info("No se proporcionó NEWS_API_KEY. Se omite el factor de noticias.")
        return []

    params = {
        "apiKey": api_key,
        "q": query,
        "language": language,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "sortBy": "publishedAt",
        "pageSize": 100,
    }

    logging.info("Descargando noticias económicas entre %s y %s", params["from"], params["to"])
    try:
        response = requests.get(NEWS_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - depende de la red.
        logging.warning("No fue posible descargar noticias: %s", exc)
        return []

    payload = response.json()
    if payload.get("status") != "ok":
        logging.warning("Respuesta inesperada de la API de noticias: %s", json.dumps(payload)[:200])
        return []

    articles = payload.get("articles", [])
    logging.debug("Se recuperaron %d artículos", len(articles))
    return articles


def aggregate_sentiment(articles: Iterable[dict]) -> pd.Series:
    """Calcula un puntaje diario de sentimiento a partir de artículos."""

    analyzer = ensure_vader()
    sentiments = []

    for art in articles:
        published = art.get("publishedAt")
        if not published:
            continue
        try:
            published_dt = pd.to_datetime(published).date()
        except Exception:  # pragma: no cover - datos malformados.
            continue

        text_parts = [art.get("title", ""), art.get("description", ""), art.get("content", "")]
        text = " ".join(filter(None, text_parts))
        if not text:
            continue

        if analyzer is None:
            compound = 0.0
        else:
            scores = analyzer.polarity_scores(text)
            compound = scores.get("compound", 0.0)

        sentiments.append({"date": published_dt, "sentiment": compound})

    if not sentiments:
        logging.info("No se obtuvieron puntuaciones de sentimiento; se utilizará valor neutral.")
        return pd.Series(dtype=float)

    df = pd.DataFrame(sentiments)
    daily = df.groupby("date")["sentiment"].mean()
    logging.debug("Serie diaria de sentimiento con %d observaciones", len(daily))
    return daily


def _feature_columns(dataset: pd.DataFrame) -> List[str]:
    """Devuelve la lista ordenada de columnas explicativas."""

    lag_cols = sorted(
        (col for col in dataset.columns if col.startswith("lag_")),
        key=lambda x: int(x.split("_")[1]),
    )
    features = list(lag_cols)
    if "sentiment" in dataset.columns:
        features.append("sentiment")
    return features


def prepare_dataset(prices: pd.DataFrame, sentiment: pd.Series, lags: int = 5) -> pd.DataFrame:
    """Construye un ``DataFrame`` con variables rezagadas y sentimiento."""

    df = prices.copy()
    df["return"] = df["close"].pct_change()

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)

    if not sentiment.empty:
        sentiment_idx = pd.to_datetime(sentiment.index)
        sentiment = pd.Series(sentiment.values, index=sentiment_idx, name="sentiment")
        df = df.join(sentiment, how="left")
        df["sentiment"].fillna(method="ffill", inplace=True)
        df["sentiment"].fillna(0.0, inplace=True)
    else:
        df["sentiment"] = 0.0

    df["target"] = df["return"].shift(-1)
    df.dropna(inplace=True)
    logging.debug("Conjunto de datos preparado con %d filas", len(df))
    return df


def fit_econometric_model(dataset: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Ajusta un modelo OLS con variables rezagadas y sentimiento."""

    features = _feature_columns(dataset)
    X = dataset[features]
    X = sm.add_constant(X)
    y = dataset["target"]

    model = sm.OLS(y, X, missing="drop")
    results = model.fit()
    logging.info("Modelo OLS ajustado con R^2 de %.3f", results.rsquared)
    return results


def forecast_returns(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    dataset: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Genera pronósticos iterativos de rendimientos para el horizonte dado."""

    if horizon <= 0:
        raise ValueError("El horizonte debe ser positivo.")

    last_row = dataset.iloc[-1]
    features = _feature_columns(dataset)
    lag_cols = [col for col in features if col.startswith("lag_")]
    lag_values = [last_row[col] for col in lag_cols]
    sentiment_value = float(last_row.get("sentiment", 0.0))

    predictions = []
    for _ in range(horizon):
        exog = {name: 0.0 for name in results.model.exog_names}
        exog["const"] = 1.0
        for col, value in zip(lag_cols, lag_values):
            exog[col] = value
        if "sentiment" in exog:
            exog["sentiment"] = sentiment_value

        exog_df = pd.DataFrame([exog])
        pred = float(results.predict(exog_df)[0])
        predictions.append(pred)

        # Actualizamos los rezagos para la siguiente iteración.
        lag_values = [pred] + lag_values[:-1]

    index = pd.date_range(dataset.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="B")
    return pd.Series(predictions, index=index, name="forecast_return")


def build_forecast(prices: pd.DataFrame, predicted_returns: pd.Series) -> pd.DataFrame:
    """Transforma rendimientos pronosticados en precios esperados."""

    last_price = prices["close"].iloc[-1]
    forecasts = []
    current_price = last_price
    for date, ret in predicted_returns.items():
        current_price *= (1 + ret)
        forecasts.append({"date": date, "expected_return": ret, "expected_price": current_price})

    forecast_df = pd.DataFrame(forecasts).set_index("date")
    return forecast_df


def run_pipeline(args: argparse.Namespace) -> PredictionResult:
    """Orquesta el flujo completo de descarga, entrenamiento y pronóstico."""

    prices = fetch_gold_prices(ticker=args.ticker)
    news_articles = fetch_financial_news(
        api_key=os.environ.get("NEWS_API_KEY"),
        from_date=dt.date.today() - dt.timedelta(days=args.news_window),
        to_date=dt.date.today(),
        language=args.news_language,
        query=args.news_query,
    )
    sentiment_series = aggregate_sentiment(news_articles)
    dataset = prepare_dataset(prices, sentiment_series, lags=args.lags)

    train_size = int(len(dataset) * (1 - args.test_size))
    train_dataset = dataset.iloc[:train_size]
    test_dataset = dataset.iloc[train_size:]

    model = fit_econometric_model(train_dataset)

    if not test_dataset.empty:
        features = _feature_columns(train_dataset)
        X_test = sm.add_constant(test_dataset[features], has_constant="add")
        y_test = test_dataset["target"]
        preds_test = model.predict(X_test)
        mse = float(((preds_test - y_test) ** 2).mean())
        logging.info("Error cuadrático medio en el tramo de prueba: %.6f", mse)

    predicted_returns = forecast_returns(model, dataset, args.horizon)
    forecast_df = build_forecast(prices, predicted_returns)

    return PredictionResult(
        forecast=forecast_df,
        model_summary=model.summary().as_text(),
        sentiment_used=not sentiment_series.empty,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Define y analiza los parámetros de la línea de comandos."""

    parser = argparse.ArgumentParser(description="Pronóstico econométrico del precio del oro")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker de yfinance a utilizar")
    parser.add_argument("--horizon", type=int, default=5, help="Horizonte de pronóstico en días hábiles")
    parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="Cantidad de rezagos diarios de rendimiento a incluir en el modelo",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de observaciones reservadas para validación (0-0.5)",
    )
    parser.add_argument(
        "--news-window",
        type=int,
        default=14,
        help="Días hacia atrás para recopilar noticias económicas",
    )
    parser.add_argument("--news-language", default="es", help="Idioma de las noticias a consultar")
    parser.add_argument(
        "--news-query",
        default="oro AND (economia OR inflación OR dólar)",
        help="Consulta booleana para NewsAPI",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Ruta opcional para exportar los pronósticos en CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilita salida detallada de depuración",
    )

    args = parser.parse_args(argv)
    if not 0 < args.test_size < 0.5:
        parser.error("--test-size debe estar entre 0 y 0.5")
    if args.lags <= 0:
        parser.error("--lags debe ser positivo")

    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Punto de entrada principal."""

    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        result = run_pipeline(args)
    except Exception as exc:  # pragma: no cover - salida controlada para CLI.
        logging.error("Se produjo un error en la ejecución: %s", exc)
        raise SystemExit(1) from exc

    logging.info("Pronóstico generado para %d días", len(result.forecast))
    if args.export:
        result.forecast.to_csv(args.export)
        logging.info("Resultados exportados a %s", args.export)

    print("\n=== Pronósticos de la onza de oro ===")
    print(result.forecast.to_string(float_format=lambda x: f"{x:0.4f}"))
    print("\n=== Resumen del modelo econométrico ===")
    print(result.model_summary)
    print(
        "\nFactor de noticias utilizado:" if result.sentiment_used else "\nNo se incorporó factor de noticias."
    )


if __name__ == "__main__":
    main()

