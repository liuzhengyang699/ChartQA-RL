# ChartQA path config

`paths.example.json` is the committed template for model, data, cache, and output directories.

Before running training or evaluation:

```bash
cp config/paths.example.json config/paths.json
```

Then edit `config/paths.json` to match your own machine. The local `config/paths.json` file is ignored by git.

If you want to keep the config somewhere else, set:

```bash
export CHARTQA_PATH_CONFIG=/abs/path/to/paths.json
```
