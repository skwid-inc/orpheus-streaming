## Quick Example

```bash
# Generate 1000 (500*2) test samples
python generate_audio.py --use-fixed-text --num-items 500

# Evaluate them
python eval_with_gemini.py --input-dir outputs/tts_download/orpheus
```


# Hallucination Evaluation Scripts

Scripts for evaluating TTS audio quality and detecting hallucinations using Google's Gemini model.

## Prerequisites

1. **Orpheus TTS Server**: Ensure the server is running on localhost:9090
   ```bash
   curl http://localhost:9090/v1/voices
   ```

2. **Python Environment**: Use the `trt_new` conda environment
   ```bash
   conda activate trt_new
   ```

3. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Gemini API Key**: The script uses the `.env` file from the parent directory:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Step 1: Generate TTS Audio

```bash
# Generate audio from fixed test text
python generate_audio.py \
  --use-fixed-text \
  --num-items 100 \
  --samples-per-prompt 2 \
  --concurrency 16
```

Options:
- `--base-url`: TTS server URL (default: http://localhost:9090)
- `--input`: Path to JSONL file with prompts
- `--text-key`: Field name for text in JSONL (auto-detect if omitted)
- `--use-fixed-text`: Use hardcoded test text
- `--fixed-text`: Override the default fixed text
- `--num-items`: Number of items when using fixed text (default: 500)
- `--samples-per-prompt`: Number of audio samples per text (default: 2)
- `--concurrency`: Number of parallel requests (default: 8)

### Step 2: Evaluate with Gemini

```bash
python eval_with_gemini.py \
  --input-dir outputs/tts_download/orpheus \
  --concurrency 100
```

Options:
- `--input-dir`: Folder with audio files to evaluate
- `--output-dir`: Where to save results (default: outputs/hallucinations_eval)
- `--concurrency`: Number of parallel evaluations (default: 100)
- `--eval-retries`: Number of retries on failure (default: 1)

## Output Structure

Generated audio:
```
outputs/tts_download/orpheus/
  item_00000/
    text.txt        # Reference text
    sample_1.wav    # Generated audio (24kHz, 16-bit, mono)
    sample_2.wav
```

Evaluation results:
```
outputs/hallucinations_eval/
  summary.json      # Aggregated statistics
  details.jsonl     # Detailed results for problematic samples
  samples/          # Audio files with detected issues
```

## Evaluation Metrics

- **hallucinations_level**: 
  - `none`: Exact match (numbers/symbols may be verbalized)
  - `medium`: 1-2 word insertions or minor repetitions
  - `high`: 3+ word additions, gibberish, or wrong content