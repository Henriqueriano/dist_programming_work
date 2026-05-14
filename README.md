# dist_programming_work

Compara o processamento de dados de vacinação COVID-19 (OpenDataSUS) usando Apache Spark e Pandas, gerando um dashboard com os resultados e o tempo de cada abordagem.

## Download dos dados

```bash
curl -L -H "Range: bytes=0-<N>" \
  "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIPNI/COVID/completo/part-00000-3e186cda-f0ce-4a5c-89e1-3355ebc57515-c000.csv" \
  -o vacinas.csv
```

Substitua `<N>` pelo limite em bytes. **Não baixe mais do que 1/3 da sua RAM disponível** — o Pandas precisa carregar o arquivo inteiro na memória, e ultrapassar esse limite vai forçar uso de swap, tornando a comparação injusta.

Exemplos:

| RAM disponível | Limite recomendado | `<N>`        |
|---------------:|-------------------:|:-------------|
| 32 GB          | ~10 GB             | 10737418239  |
| 16 GB          | ~5 GB              | 5368709119   |
| 8 GB           | ~2,5 GB            | 2684354559   |

## Instalação

```bash
uv sync
```

## Uso

```bash
uv run reader.py vacinas.csv
```

O script processa o arquivo duas vezes (Spark e Pandas) e salva o dashboard em `dashboard.png`.
