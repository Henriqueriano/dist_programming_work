import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display; necessário em terminal puro
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

STATE_REGIONS = {
    "AC": "Norte",       "AM": "Norte",       "AP": "Norte",   "PA": "Norte",
    "RO": "Norte",       "RR": "Norte",       "TO": "Norte",
    "AL": "Nordeste",    "BA": "Nordeste",    "CE": "Nordeste","MA": "Nordeste",
    "PB": "Nordeste",    "PE": "Nordeste",    "PI": "Nordeste","RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste","GO": "Centro-Oeste","MS": "Centro-Oeste","MT": "Centro-Oeste",
    "ES": "Sudeste",     "MG": "Sudeste",     "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul",         "RS": "Sul",         "SC": "Sul",
}

REGION_COLORS = {
    "Sudeste":      "#0891B2",
    "Nordeste":     "#10B981",
    "Sul":          "#8B5CF6",
    "Norte":        "#F59E0B",
    "Centro-Oeste": "#EC4899",
}

# ── Configurações do Spark ────────────────────────────────────────────────────
WORKERS            = "*"    # local[*] = todas as threads; ou ex: "10"
PARTITIONS         = 24     # repartition — recomendado: 2× número de cores
SHUFFLE_PARTITIONS = 24     # spark.sql.shuffle.partitions — idem
DRIVER_MEMORY      = "16g"  # heap da JVM; aumente se OOM no cache()
TOP_N              = 3      # top N vacinas por estado


@dataclass
class Timing:
    # Armazena o tempo de cada etapa de processamento
    read:      float = 0.0
    transform: float = 0.0
    aggregate: float = 0.0

    @property
    def total(self) -> float:
        return self.read + self.transform + self.aggregate


def read_csv_columns(csv_path: str) -> list[str]:
    # Lê apenas o header do CSV para detectar os nomes das colunas.
    # Se for glob, usa o primeiro arquivo que casar.
    import glob as _glob
    matches = _glob.glob(csv_path)
    path = matches[0] if matches else csv_path
    with open(path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f, delimiter=";"))


def _detect_column(columns: list[str], *keywords: str) -> str:
    # Retorna a primeira coluna cujo nome contenha todas as keywords fornecidas
    normalized = [c.strip('"').lower() for c in columns]
    for col, norm in zip(columns, normalized):
        if all(kw in norm for kw in keywords):
            return col.strip('"')
    return columns[0].strip('"')


# ── Spark ─────────────────────────────────────────────────────────────────────

def process_with_spark(
    csv_path: str, state_col: str, vaccine_col: str
) -> tuple[pd.DataFrame, Timing]:
    # Cria a SparkSession em modo local usando todas as threads disponíveis.
    # local[N] simula um cluster de N workers na mesma máquina.
    spark = (
        SparkSession.builder
        .appName("COVID19")
        .master(f"local[{WORKERS}]")
        .config("spark.driver.memory",          DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    timing = Timing()

    # Leitura: o Spark aceita path único, glob ("part-*.csv") ou diretório.
    # .select() projeta apenas as colunas necessárias antes de cachear,
    # reduzindo o volume de dados mantido em memória.
    # cache() + count() forçam a execução lazy e materializam o DataFrame em RAM.
    t = time.perf_counter()
    df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .csv(csv_path)
        .select(state_col, vaccine_col)
    )
    df.cache()
    df.count()  # ação que dispara a leitura real do arquivo
    timing.read = time.perf_counter() - t

    # Transformação: repartition() redistribui os dados entre os workers
    # (shuffle), garantindo balanceamento para o groupBy seguinte.
    # Sem repartition, todos os dados de um mesmo estado poderiam ficar
    # em uma única partição, sobrecarregando apenas um worker.
    t = time.perf_counter()
    df_clean = (
        df.repartition(PARTITIONS)
        .filter(F.col(state_col).isNotNull() & F.col(vaccine_col).isNotNull())
        .withColumn("state",   F.upper(F.col(state_col)))
        .withColumn("vaccine", F.col(vaccine_col))
        .select("state", "vaccine")
    )
    df_clean.cache()
    df_clean.count()  # força execução das transformações acima
    timing.transform = time.perf_counter() - t

    # Agrupamento: conta por (estado, vacina) e aplica window function para
    # rankear as vacinas dentro de cada estado sem precisar de um segundo groupBy.
    # Window.partitionBy divide os dados por estado; cada partição é ordenada
    # independentemente pelos workers em paralelo — operação cara (shuffle).
    t = time.perf_counter()
    counts = (
        df_clean.groupBy("state", "vaccine")
        .agg(F.count("*").alias("count"))
    )
    window = Window.partitionBy("state").orderBy(F.desc("count"))
    result = (
        counts
        .withColumn("rank", F.rank().over(window))  # rank por estado, 1 = mais aplicada
        .filter(F.col("rank") <= TOP_N)
        .orderBy("state", "rank")
        .toPandas()  # traz o resultado final para o processo Python
    )
    timing.aggregate = time.perf_counter() - t

    spark.stop()
    return result, timing


# ── Pandas ────────────────────────────────────────────────────────────────────

def process_with_pandas(
    csv_path: str, state_col: str, vaccine_col: str
) -> tuple[pd.DataFrame, Timing]:
    # usecols faz o parser CSV ignorar as outras 30 colunas já na leitura,
    # reduzindo memória e tempo de I/O significativamente.
    timing = Timing()

    t = time.perf_counter()
    df = pd.read_csv(
        csv_path, sep=";", encoding="utf-8",
        usecols=[state_col, vaccine_col], low_memory=False,
    )
    timing.read = time.perf_counter() - t

    t = time.perf_counter()
    df_clean = df.dropna(subset=[state_col, vaccine_col]).copy()
    df_clean["state"]   = df_clean[state_col].str.upper()
    df_clean["vaccine"] = df_clean[vaccine_col]
    timing.transform = time.perf_counter() - t

    # groupby().head(TOP_N) após sort garante os N maiores por estado
    t = time.perf_counter()
    counts = (
        df_clean.groupby(["state", "vaccine"]).size()
        .reset_index(name="count")
    )
    result = (
        counts.sort_values("count", ascending=False)
        .groupby("state").head(TOP_N)
        .sort_values(["state", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    timing.aggregate = time.perf_counter() - t

    return result, timing


# ── Python puro ───────────────────────────────────────────────────────────────

def process_with_python(
    csv_path: str, state_col: str, vaccine_col: str
) -> tuple[pd.DataFrame, Timing]:
    # Implementação sem bibliotecas externas: csv da stdlib + dict + list.
    # Serve como baseline para comparar o custo real de cada abstração.
    timing = Timing()

    # Leitura: extrai apenas as duas colunas necessárias em uma list comprehension,
    # carregando os pares (estado, vacina) de todas as linhas na memória.
    t = time.perf_counter()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header      = [c.strip('"') for c in next(reader)]
        state_idx   = header.index(state_col)
        vaccine_idx = header.index(vaccine_col)
        raw = [(row[state_idx], row[vaccine_idx]) for row in reader]
    timing.read = time.perf_counter() - t

    t = time.perf_counter()
    pairs = [
        (s.strip('"').upper(), v.strip('"'))
        for s, v in raw
        if s.strip('"') and v.strip('"')
    ]
    timing.transform = time.perf_counter() - t

    # Agrupamento em duas passagens: primeiro conta por (estado, vacina),
    # depois agrupa por estado e ordena para extrair o top N de cada um.
    t = time.perf_counter()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for state, vaccine in pairs:
        counts[(state, vaccine)] += 1

    state_vaccines: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (state, vaccine), n in counts.items():
        state_vaccines[state].append((vaccine, n))

    rows = []
    for state, vaccines in sorted(state_vaccines.items()):
        for vaccine, n in sorted(vaccines, key=lambda x: -x[1])[:TOP_N]:
            rows.append({"state": state, "vaccine": vaccine, "count": n})

    result = pd.DataFrame(rows)
    timing.aggregate = time.perf_counter() - t

    return result, timing


# ── Gráficos ──────────────────────────────────────────────────────────────────

ENGINE_COLORS = {
    "Spark":       "#0891B2",
    "Pandas":      "#10B981",
    "Python puro": "#F59E0B",
}


def _style_axis(ax) -> None:
    # Aplica o tema escuro padrão em todos os eixos
    ax.set_facecolor("#0D2137")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#1E3A5F")
    ax.spines["left"].set_color("#1E3A5F")


def generate_results_chart(result: pd.DataFrame) -> None:
    # Gera dashboard_results.png com a vacina mais aplicada por estado (barras)
    # e a distribuição de doses por região (pizza).
    fig, (ax_top, ax_pie) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor("#0A1628")
    _style_axis(ax_top)
    _style_axis(ax_pie)

    # Filtra apenas a vacina de rank 1 por estado para o gráfico de barras
    top1 = (
        result[result.groupby("state")["count"].transform("max") == result["count"]]
        .drop_duplicates("state")
        .sort_values("count", ascending=False)
        .head(20)
        .copy()
    )
    top1["region"] = top1["state"].map(STATE_REGIONS)

    bar_colors = [REGION_COLORS.get(r, "#94A3B8") for r in top1["region"]]
    ax_top.barh(top1["state"] + " — " + top1["vaccine"], top1["count"],
                color=bar_colors, edgecolor="#0A1628", linewidth=0.5)
    ax_top.set_title("Vacina mais aplicada por Estado (top 20)", color="white", fontsize=13, pad=12)
    ax_top.set_xlabel("Total de Doses", color="#94A3B8", fontsize=10)
    ax_top.tick_params(colors="white", labelsize=7)
    ax_top.xaxis.grid(True, color="#1E3A5F", linestyle="--", alpha=0.5)
    ax_top.set_axisbelow(True)
    ax_top.invert_yaxis()
    legend_patches = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
    ax_top.legend(handles=legend_patches, loc="lower right",
                  facecolor="#122338", labelcolor="white", fontsize=8)

    all_top1 = (
        result[result.groupby("state")["count"].transform("max") == result["count"]]
        .drop_duplicates("state")
        .copy()
    )
    all_top1["region"] = all_top1["state"].map(STATE_REGIONS)
    by_region  = all_top1.groupby("region")["count"].sum().sort_values(ascending=False)
    pie_colors = [REGION_COLORS.get(r, "#94A3B8") for r in by_region.index]
    _, _, autotexts = ax_pie.pie(
        by_region, labels=by_region.index, autopct="%1.1f%%",
        colors=pie_colors, startangle=140,
        textprops={"color": "white", "fontsize": 10},
        wedgeprops={"edgecolor": "#0A1628", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)
    ax_pie.set_title("Doses da vacina top-1 por Região", color="white", fontsize=13, pad=12)

    plt.suptitle("Apache Spark — COVID-19 OpenDataSUS",
                 color="#22D3EE", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("dashboard_results.png", dpi=150, bbox_inches="tight", facecolor="#0A1628")
    print("dashboard_results.png saved")


def generate_timing_chart(timings: dict[str, Timing]) -> None:
    # Gera dashboard_timing.png com barras agrupadas por etapa para cada engine.
    # Funciona com qualquer número de engines (1 com --spark, 3 no modo completo).
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0A1628")
    _style_axis(ax)

    labels  = ["Leitura", "Transformação", "Agrupamento", "Total"]
    n       = len(timings)
    w       = 0.7 / n
    # centraliza os grupos de barras em cada posição do eixo x
    offsets = [((i - (n - 1) / 2) * w) for i in range(n)]

    for (engine, timing), offset in zip(timings.items(), offsets):
        vals = [timing.read, timing.transform, timing.aggregate, timing.total]
        bars = ax.bar(
            [x + offset for x in range(len(labels))], vals, w,
            label=engine, color=ENGINE_COLORS.get(engine, "#94A3B8"),
        )
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h * 1.02,
                    f"{h:.1f}s", ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color="white", fontsize=11)
    title = " vs ".join(timings.keys()) + " — Tempo por Etapa (s)"
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_ylabel("Segundos", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="white")
    ax.yaxis.grid(True, color="#1E3A5F", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#122338", labelcolor="white", fontsize=10)

    plt.suptitle("Apache Spark — COVID-19 OpenDataSUS",
                 color="#22D3EE", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("dashboard_timing.png", dpi=150, bbox_inches="tight", facecolor="#0A1628")
    print("dashboard_timing.png saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def print_timing_summary(timings: dict[str, Timing]) -> None:
    # Imprime tabela comparativa de tempo por etapa com o vencedor de cada linha
    steps = [("Read", "read"), ("Transform", "transform"), ("Aggregate", "aggregate"), ("Total", "total")]
    engines = list(timings.keys())
    header = f"  {'Step':<12}" + "".join(f"  {e:>12}" for e in engines) + "   Winner"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for step_name, attr in steps:
        vals   = {e: getattr(t, attr) for e, t in timings.items()}
        winner = min(vals, key=vals.__getitem__)
        row    = f"  {step_name:<12}" + "".join(f"  {v:>10.2f}s" for v in vals.values())
        print(f"{row}   {winner}")


if __name__ == "__main__":
    args = sys.argv[1:]
    spark_only = "--spark" in args
    args = [a for a in args if a != "--spark"]

    if len(args) != 1:
        print(f"Usage: uv run {sys.argv[0]} [--spark] <file.csv>")
        sys.exit(1)

    csv_path = args[0]

    columns     = read_csv_columns(csv_path)
    state_col   = _detect_column(columns, "uf")
    vaccine_col = _detect_column(columns, "vacina", "nome")

    print(f"state column  : {state_col}")
    print(f"vaccine column: {vaccine_col}")

    timings: dict[str, Timing] = {}

    spark_result, spark_timing = process_with_spark(csv_path, state_col, vaccine_col)
    timings["Spark"] = spark_timing
    print(f"\n[spark]  {len(spark_result)} rows — total {spark_timing.total:.2f}s")
    print(spark_result.head(10).to_string(index=False))

    if not spark_only:
        pandas_result, pandas_timing = process_with_pandas(csv_path, state_col, vaccine_col)
        timings["Pandas"] = pandas_timing
        print(f"\n[pandas] {len(pandas_result)} rows — total {pandas_timing.total:.2f}s")
        print(pandas_result.head(10).to_string(index=False))

        python_result, python_timing = process_with_python(csv_path, state_col, vaccine_col)
        timings["Python puro"] = python_timing
        print(f"\n[python] {len(python_result)} rows — total {python_timing.total:.2f}s")
        print(python_result.head(10).to_string(index=False))

    print("\nTiming summary:")
    print_timing_summary(timings)

    generate_results_chart(spark_result)
    generate_timing_chart(timings)
