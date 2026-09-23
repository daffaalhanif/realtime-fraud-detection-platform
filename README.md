# Realtime Fraud Detection Platform

A real-time risk scoring platform for card-not-present payment transactions. Every incoming transaction is evaluated through a two-tier model pipeline (a classic model for all transactions, a custom PyTorch Transformer for borderline cases only), resulting in one of three decision classes: decline, approve with review flag, or approve normally. Built on the public IEEE-CIS Fraud Detection dataset.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker Desktop (or Docker Engine with the Compose plugin)
- A Kaggle account

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd realtime-fraud-detection-platform
```

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv` and installs all dependencies declared in `pyproject.toml`.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in the values in `.env`. `SCORING_API_KEY` and `JWT_SECRET_KEY` should be random strings, for example generated with:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Download the dataset

The raw dataset is not included in this repository and must be downloaded yourself through the Kaggle API.

1. Log in to [Kaggle](https://www.kaggle.com), go to your account settings, and create a new API token. This downloads a `kaggle.json` file.
2. Place that file at `~/.kaggle/kaggle.json`.
3. On the [IEEE-CIS Fraud Detection competition page](https://www.kaggle.com/competitions/ieee-fraud-detection), accept the competition rules. The API download will fail until this step is done.
4. Download and extract the dataset into a local `data/` folder:

```bash
uvx kaggle competitions download -c ieee-fraud-detection -p data/
unzip data/ieee-fraud-detection.zip -d data/
```

The `data/` folder is excluded from version control and must never be committed.

### 5. Start local infrastructure

```bash
docker compose up -d
```

This starts Redis, Postgres, and Kafka.
