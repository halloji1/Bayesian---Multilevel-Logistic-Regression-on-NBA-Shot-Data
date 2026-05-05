# Kaggle Credentials Setup

This project downloads NBA shot-log data from Kaggle via the `kaggle` and `kagglehub`
Python packages. Both require a valid API token.

## 1. Create / obtain your Kaggle API token

1. Log in at <https://www.kaggle.com>.
2. Click your profile avatar → **Settings** → **Account**.
3. Scroll to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will be downloaded. It looks like:
   ```json
   {"username": "your_username", "key": "your_api_key"}
   ```

## 2. Place the token where the Kaggle client expects it

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The `chmod 600` step is required — the client refuses to run if the file is
world-readable.

## 3. Verify the setup

```bash
kaggle datasets list --search "nba shot"
```

You should see a list of matching datasets with no authentication errors.

Alternatively, from Python:

```python
import kaggle
kaggle.api.authenticate()   # raises if credentials are missing or invalid
print("Kaggle authentication OK")
```

## 4. Troubleshooting

| Error | Fix |
|-------|-----|
| `OSError: Could not find kaggle.json` | Check the file is at `~/.kaggle/kaggle.json` |
| `401 Unauthorized` | Regenerate the token on the Kaggle website |
| `Permission denied` | Re-run `chmod 600 ~/.kaggle/kaggle.json` |
| `kaggle: command not found` | Activate the project virtual environment first |
