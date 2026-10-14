# Beveridge

**System One decision models with open weights.** Typed yes/no, choice and score decisions over any text, with a
calibrated probability for every option, without generating text, on your own machines.

This package is the same as [`bev-decisions`](https://pypi.org/project/bev-decisions/), under the models' name:

```bash
pip install beveridge            # installs bev-decisions; Beveridge-Large-C runs on a laptop CPU or any GPU
pip install "beveridge[pro]"     # adds Beveridge-Pro-C: one GPU with about 60 GB
```

```python
import beveridge

model = beveridge.load()
answers = model.decide(
    "Subject: Charged twice for order 88412. Please refund one payment today.",
    {"urgent": {"type": "noul", "instructions": "Does the customer ask for same-day handling?"}},
)
```

`import bev` works too. Documentation, benchmarks and examples:
[github.com/sathiskumarjothi0-oss/bev-decisions-pypi](https://github.com/sathiskumarjothi0-oss/bev-decisions-pypi) ·
models: [Beveridge-Large-C](https://huggingface.co/sathiskumarjothi0/Beveridge-Large-C),
[Beveridge-Pro-C](https://huggingface.co/sathiskumarjothi0/Beveridge-Pro-C) · [bev.sjothi.com](https://bev.sjothi.com)

## About the name

Beveridge is named after Sir William Beveridge (1879–1963), the British economist, born in British India and educated
in England, whose 1942 Beveridge Report laid the foundations of the modern welfare state and of the National Health
Service (NHS), built for the good of future generations. The name is a reminder of what we want from the model:
dependable, fair decisions that serve people, made carefully and checked against the facts.

## Licence

Apache-2.0, © 2026 Dr. Sathiskumar JOTHI.
