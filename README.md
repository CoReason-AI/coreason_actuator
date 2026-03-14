# coreason_actuator

The sandboxed physical execution mechanism and Pearlian Do-Operator

[![CI/CD](https://github.com/CoReason-AI/coreason_actuator/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_actuator/actions/workflows/ci-cd.yml)
[![PyPI](https://img.shields.io/pypi/v/coreason_actuator.svg)](https://pypi.org/project/coreason_actuator/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/coreason_actuator.svg)](https://pypi.org/project/coreason_actuator/)
[![License](https://img.shields.io/github/license/CoReason-AI/coreason_actuator)](https://github.com/CoReason-AI/coreason_actuator/blob/main/LICENSE)
[![Codecov](https://codecov.io/gh/CoReason-AI/coreason_actuator/branch/main/graph/badge.svg)](https://codecov.io/gh/CoReason-AI/coreason_actuator)
[![Downloads](https://static.pepy.tech/badge/coreason_actuator)](https://pepy.tech/project/coreason_actuator)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Getting Started

### Prerequisites

- Python 3.14+
- uv

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_actuator.git
    cd coreason_actuator
    ```
2.  Install dependencies:
    ```sh
    uv sync --all-extras --dev
    ```

### Usage

-   Run the linter:
    ```sh
    uv run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    uv run pytest
    ```
