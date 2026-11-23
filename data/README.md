# Data preparation

Place the evaluation datasets here, or configure `DATA_DIR` in `local_config.py`.

| Default filename | Source |
|---|---|
| `locomo_10.json` | [LoCoMo](https://github.com/snap-research/locomo), `data/locomo10.json` |
| `longmemeval_s.json` | [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval) |
| Optional cleaned LongMemEval | [Cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |

The optional `python data/download_data.py` obtains the default datasets and
prints their SHA-256 hashes. Existing files are not overwritten.
`reference_manifest.json` contains reference dataset hashes.
For a different snapshot, use the expected filename under the selected `DATA_DIR`.

Loader schemas:

- LoCoMo: JSON array with `conversation` and `qa` in each row.
- LongMemEval: JSON array with `question_id`, `question`, `answer`, `question_type`,
  `question_date`, `haystack_sessions` and `haystack_dates`.

## Training corpus

Prepare `data/dailydialog/train.json` and `data/dailydialog/validation.json`.
Each is a JSON array whose rows contain an `utterances` or `dialog` list:

```json
[{"utterances": ["Hello.", "How are you?", "I am fine."]}]
```

The training loader uses two preceding turns to predict the next utterance,
padding or truncating source and target sequences to 128 tokens.
