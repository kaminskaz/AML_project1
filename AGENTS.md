# AGENTS.md

## Project Overview

Machine Learning project using Python with scikit-learn and custom implementations.

## Environment Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
.
├── .venv/
├── src/
│   ├── __init__.py
│   ├── ...
├── data/
├── requirements.txt
└── README.md
```

## Dependencies

Core packages managed in `requirements.txt`:
- scikit-learn
- pandas
- numpy


## Code Style

- Clean, readable code preferred over comments
- Use docstrings for functions and classes
- Type hints encouraged
- Follow PEP 8

## Custom Implementations

Parts of the project implemented from scratch (not sklearn):
- Custom algorithms in `src/models/`


