# HuggingFace Authentication Setup

Some models in the registry (like `ai4bharat/indic-bert`) are gated and require authentication. This guide shows you how to set up HuggingFace authentication.

## Step 1: Get Your HuggingFace Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Choose:
   - **Name**: `indic-tokenizer-lab` (or any name you prefer)
   - **Type**: **Read** (sufficient for downloading models)
4. Click **"Generate token"**
5. **Copy the token** immediately (you won't be able to see it again!)

The token will look like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## Step 2: Configure Authentication

You have three options:

### Option A: Login via CLI (Recommended)

```bash
pip install huggingface_hub
huggingface-cli login
```

When prompted, paste your token. This will save it to `~/.huggingface/token`.

### Option B: Environment Variable

```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent:

```bash
echo 'export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### Option C: Use Token in Code

The code will automatically use the token if you've logged in via CLI or set the environment variable. If you need to pass it explicitly, you can modify the code (see below).

## Step 3: Accept Model License (for Gated Models)

For gated models like `ai4bharat/indic-bert`:

1. Go to the model page: https://huggingface.co/ai4bharat/indic-bert
2. Click **"Agree and access repository"** to accept the license
3. This is a one-time step per model

## Step 4: Test Authentication

```bash
# Test that authentication works
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert'); print('✓ Authentication successful!')"
```

## Troubleshooting

### Error: "401 Client Error: Unauthorized"

- Make sure you've accepted the model license on HuggingFace
- Verify your token is correct: `huggingface-cli whoami`
- Try logging in again: `huggingface-cli login`

### Error: "Repository not found"

- Make sure you've accepted the model license
- Check that the model name is correct in `tokenizers/registry.yaml`

### Token Not Working

- Generate a new token if needed
- Make sure the token has **Read** permissions
- Check that you're logged in: `huggingface-cli whoami`

## Security Notes

- **Never commit your token to git** - it's already in `.gitignore`
- Tokens are stored in `~/.huggingface/token` (local file, not in repo)
- Use environment variables in production/CI environments
- Rotate tokens periodically for security

## CI/CD Setup

For GitHub Actions or other CI, add the token as a secret:

```yaml
# .github/workflows/ci.yml
env:
  HUGGING_FACE_HUB_TOKEN: ${{ secrets.HUGGING_FACE_HUB_TOKEN }}
```

Then add `HUGGING_FACE_HUB_TOKEN` as a repository secret in GitHub Settings → Secrets.

