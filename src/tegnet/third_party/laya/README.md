# Laya Integration

This directory contains Laya components modified for TEGNet, based on [Laya](https://github.com/NandhaKishorM/laya/tree/9d955671415fc19f069b9cc998928075c1f255ec), revision `9d955671415fc19f069b9cc998928075c1f255ec`, and licensed under [Apache-2.0](LICENSE).

`common.py` provides decision scoring, checkpoint-configuration support, and choice-question batching. [decision.py](../../decision.py) assembles complete reject/keep inputs, validates their length before model inference, and derives edge weights from a softmax of the raw option logits.

The model preserves the parameter names and shapes required to load Laya checkpoints.
