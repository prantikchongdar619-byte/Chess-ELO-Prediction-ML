# ♟️ Chess ELO Rating Prediction

A machine learning project that predicts chess player ELO ratings (White & Black) from game data using deep neural networks and advanced feature engineering.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Future Improvements](#future-improvements)

---

## Overview

This project uses chess game data in PGN (Portable Game Notation) format to predict the ELO ratings of both players in a game. ELO is a standardized skill rating system widely used in chess. The model is trained on 25,000 games and evaluated on a separate test set.

**Targets:** `WhiteElo`, `BlackElo`

---

## Dataset

**Source:** [Kaggle — Finding ELO](https://www.kaggle.com/competitions/finding-elo)

| Property | Details |
|---|---|
| Total rows | 50,000 |
| Total columns | 11 (9 features + 2 targets) |
| Format | PGN + CSV (Stockfish evaluations) |
| Train / Test split | 25,000 / 25,000 |

**Files used:**
- `data.pgn` — Chess games in PGN format
- `stockfish.csv` — Per-move Stockfish engine evaluations
- `sampleSubmission.csv` — Submission format reference

---

## Project Structure

```
Chess-ELO-Prediction-ML/
│
├── ELO.ipynb               # Main notebook with full pipeline
├── finding-elo.zip         # Raw dataset (Kaggle)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Feature Engineering

Raw PGN data is parsed and transformed into a rich feature set:

### Game-Level Features
| Feature | Description |
|---|---|
| `move_count` | Total number of moves in the game |
| `move_count_squared` | Non-linear game length signal |
| `move_count_log` | Log-scaled game length |
| `game_length_cat` | Categorical bins: very short → very long |
| `Result_map` | Mapped game result: `1` (White), `0` (Draw), `-1` (Black) |
| `Elo_diff` | WhiteElo − BlackElo (train only) |

### Move-Based Features
| Feature | Description |
|---|---|
| `captures` | Number of captures (`x` in notation) |
| `checks` | Number of checks (`+`) |
| `castles_kingside` / `castles_queenside` | Castling frequency |
| `checkmates` | Number of checkmates (`#`) |
| `queen_moves`, `rook_moves`, `bishop_moves`, `knight_moves` | Piece activity |
| `promotions` | Pawn promotions (`=`) |
| `opening_moves` / `midgame_moves` / `closing_moves` | Move phase sequences |
| `opening_type` | Classified opening (Sicilian, French, Indian Defense, etc.) |

### Ratio Features (normalized by game length)
| Feature | Description |
|---|---|
| `captures_per_move` | Capture intensity |
| `checks_per_move` | Aggression indicator |
| `queen_activity` | Queen usage ratio |
| `tactical_density` | Combined captures + checks ratio |

### Stockfish Score Features
| Feature | Description |
|---|---|
| `white_score_sum/avg/max/std` | White advantage statistics |
| `black_score_sum/avg/max/std` | Black advantage statistics |
| `score_volatility` | Standard deviation of score swings |
| `advantage_changes` | How many times advantage flipped |
| `blunders_white/black` | Moves with eval drop > 200 centipawns |
| `brilliant_white/black` | Moves with eval gain > 300 centipawns |
| `avg_advantage` | Average positional advantage |

### Encoding & Scaling
- Categorical move sequences encoded with `LabelEncoder` (unseen labels mapped to `-1`)
- All features scaled with `StandardScaler`
- Scaler saved and reused for inverse-transforming predictions

---

## Model Architecture

A Sequential Neural Network with **Leaky ReLU** activations:

```
Input → 512 → 256 → 128 → 64 → 32 → 2 (Output)
```

Each dense layer (except the output) is followed by:
- `LeakyReLU(alpha=0.01)`
- `BatchNormalization`
- `Dropout` (0.2 for first 3 layers, 0.1 for last 2)

**Training Config:**
| Parameter | Value |
|---|---|
| Optimizer | Adam (`lr=1e-3`) |
| Loss | Mean Squared Error |
| Metric | Mean Absolute Error (MAE) |
| Epochs | 150 |
| Batch size | 64 |
| Early Stopping | patience=20, restore best weights |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=7, min_lr=1e-6) |
| Regularization | L2 (`1e-4`) on all dense layers |

---

## Results

| Metric | Value |
|---|---|
| **Overall MAE** | **174.36 ELO points** |
| WhiteElo MAE | 174.21 ELO points |
| BlackElo MAE | 174.52 ELO points |

> *Note: ELO differences of ~150–200 points correspond to roughly one skill tier in chess (e.g., intermediate → advanced). Further improvements via better feature engineering are feasible without increasing model complexity.*

