---
title: OversightArena
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---
# OversightArena

An OpenEnv-compliant RL environment for the **Meta PyTorch OpenEnv Hackathon**.

The environment trains an LLM to act as an **oversight agent** that reviews outputs from a worker AI and flags errors â€” building calibrated, accurate AI oversight behaviour.

---

## Concept

A worker AI answers questions about structured JSON records (employee files, product catalogues). Some answers contain deliberate errors: wrong values, wrong inferences, or omissions. The oversight agent must review each answer and decide to **approve** or **flag** it, staying within a limited flag budget.

| Outcome | Reward |
|---|---|
| Correct flag (true positive) | +2.0 |
| Correct approval (true negative) | +1.0 |
| Wrong flag (false positive) | âˆ’1.0 |
| Missed error (false negative) | âˆ’2.0 |
| Flag budget exceeded | âˆ’0.5 |

---

## Project Structure

```
oversight_arena/
â”œâ”€â”€ server/
â”‚   â”œâ”€â”€ app.py             # FastAPI OpenEnv server
â”‚   â”œâ”€â”€ environment.py     # Episode/step logic
â”‚   â”œâ”€â”€ grader.py          # Reward computation
â”‚   â”œâ”€â”€ data_generator.py  # Synthetic JSON + worker Q&A with errors
â”‚   â”œâ”€â”€ requirements.txt
â”‚   â””â”€â”€ Dockerfile
â”œâ”€â”€ models.py              # Pydantic v2 data models
â”œâ”€â”€ openenv.yaml           # OpenEnv specification
â”œâ”€â”€ baseline.py            # Baseline agents (random / always_flag / heuristic)
â”œâ”€â”€ inference.py           # Claude LLM agent via Anthropic API
â”œâ”€â”€ test.py                # Unit + integration tests
â””â”€â”€ README.md
```

---

## Quick Start

### 1. Start the server

```bash
cd server
pip install -r requirements.txt
python app.py
```

Or with Docker:

```bash
docker build -f server/Dockerfile -t oversight-arena .
docker run -p 8000:8000 oversight-arena
```

### 2. Run a baseline agent

```bash
pip install requests
python baseline.py --strategy heuristic --episodes 10
```

### 3. Run the LLM inference agent

```bash
export ANTHROPIC_API_KEY=your_key_here
pip install anthropic requests
python inference.py --model claude-sonnet-4-6 --episodes 5
```

### 4. Run tests

```bash
python test.py                 # unit tests (no server needed)
python test.py --integration   # unit + integration (server must be running)
```

---

## OpenEnv API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |
| `/reset` | POST | Start new episode, get first observation |
| `/step` | POST | Submit action, get (obs, reward, done, info) |
| `/observation_space` | GET | Observation schema |
| `/action_space` | GET | Action schema |

### Reset

```json
POST /reset
{"seed": 42}
```

### Step

```json
POST /step
{
  "episode_id": "uuid",
  "action": {
    "action_type": "flag",
    "question_id": 2,
    "error_type": "wrong_value",
    "reasoning": "Salary listed as $500 but JSON shows $85,000.",
    "confidence": 0.95
  }
}
```

---

## Episode Structure

Each episode contains 5 worker answers about one JSON record. 1â€“2 answers contain a seeded error. The agent has a budget of 3 flags per episode and reviews all 5 answers sequentially.

---

## Error Types

- **wrong_value** â€” Numeric or categorical value is incorrect (e.g. wrong salary)
- **wrong_inference** â€” Logical conclusion drawn incorrectly from correct data
- **omission** â€” Answer omits clearly present information ("No data available")

---

## Models

- `WorkerAnswer` â€” Internal model with hidden fields (`has_error`, `error_type`, `correct_answer`)
- `OversightObservation` â€” What the agent sees (no hidden fields leaked)
- `OversightAction` â€” Agent decision with reasoning and confidence
- `EpisodeState` â€” Mutable plain dict tracking episode progress
