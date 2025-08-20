## Quick Example

### Using Fixed Text
```bash
# Generate 1000 (500*2) test samples
python generate_audio.py --use-fixed-text --num-items 500

# Evaluate them
python eval_with_gemini.py --input-dir outputs/tts_download/orpheus
```

### Using JSONL File
```bash
# Use the provided alphanumeric.jsonl (50 prompts with numbers, dates, and codes)
# Or create your own JSONL file (each line is a JSON object with "text" field)

# Generate audio from JSONL
python generate_audio.py --input alphanumeric.jsonl --samples-per-prompt 2

# Evaluate the generated audio
python eval_with_gemini.py --input-dir outputs/tts_download/orpheus
```

The included `alphanumeric.jsonl` contains 50 challenging prompts from the [TrySalient/tts-v2-verbalized](https://huggingface.co/datasets/TrySalient/tts-v2-verbalized) dataset, featuring:
- Dollar amounts and financial figures
- Dates and times
- Phone numbers and account numbers  
- Confirmation codes with spelled-out letters
- Vehicle models and years


# Hallucination Evaluation Scripts

Scripts for evaluating TTS audio quality and detecting hallucinations using Google's Gemini model.

## Prerequisites

1. **Orpheus TTS Server**: Ensure the server is running on localhost:9090
   ```bash
   curl http://localhost:9090/v1/audio/speech/stream -X POST -H "Content-Type: application/json" -d '{"input": "test"}' --output test.wav
   ```

2. **Python Environment**: Use the `trt_new` conda environment
   ```bash
   conda activate trt_new
   ```

3. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Gemini API Key**: Add to the parent directory's `.env` file (`/workspace/orpheus-streaming/.env`):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Step 1: Generate TTS Audio

**From Fixed Text:**
```bash
python generate_audio.py \
  --use-fixed-text \
  --num-items 100 \
  --samples-per-prompt 2 \
  --concurrency 16
```

**From JSONL File:**
```bash
python generate_audio.py \
  --input alphanumeric.jsonl \
  --samples-per-prompt 2 \
  --concurrency 16
```

JSONL Format (each line is a JSON object):
```json
{"text": "Your text to synthesize here"}
{"text": "Another text prompt", "metadata": "optional fields are ignored"}
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