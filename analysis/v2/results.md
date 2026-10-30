# v2 (fullrun2000b): results

Source: `results/lambda_bench_runs/fullrun2000b/`, `fullrun2000b_loops/` and `fullrun2000b_longform/` (downloaded from GCS on 2026-10-30), compared with the v1 pickles in `results/transcribe_cache/`. Every number below comes from [`results.py`](results.py) (run it from the repo root with the project env). Computed 2026-10-30.

Setup: 2000 LibriSpeech test-clean samples x 22 corruptions x 7 models, `SEED=123`, greedy decoding, `WHISPER_LONGFORM=chunked`, image built from `main` at `5fe7566` (transformers 5.19.0, torch 2.14.1+cu126). WER is corpus WER: all substitutions, deletions and insertions over all reference words, after Whisper's English normalizer on both sides.

## 1. Data check

- **Complete**: 308,000 rows, exactly one per (sample, model, corruption). 2000 samples, 22 corruptions, 7 models. `duration_s` is present on every row (v1 lacked it on 46,200 rows).
- **4.26 h of clean speech**, 41,345 reference words per corruption and model.
- **8 samples have a clip over 30 s** (the same 8 as in v1).
- **Empty transcripts**: 134 for wav2vec2, 1 for turbo, none for the others. Same counts as v1.

## 2. v2 reproduces v1 exactly

Every one of the 308,000 transcripts is identical to v1's, character for character, for all 7 models. So every WER, CER and loop count matches v1 to the last digit.

| Model | Identical transcripts | WER v1 | WER v2 |
|---|---|---|---|
| tiny | 100% | 42.16 | 42.16 |
| base | 100% | 32.62 | 32.62 |
| small | 100% | 12.25 | 12.25 |
| medium | 100% | 7.90 | 7.90 |
| large-v3 | 100% | 5.16 | 5.16 |
| turbo | 100% | 5.68 | 5.68 |
| wav2vec2 | 100% | 20.00 | 20.00 |

What this shows:
- **The pipeline is deterministic** end to end: seeded corruptions, greedy decoding and fp16 inference give the same text on a different instance two weeks later. This holds even though v1 was split into a crashed attempt and a resume, and v2 ran in one piece.
- **The library pins match v1** in everything that affects output. Had transformers 5.21.0 been used, decoding could have differed (it already differed in memory use, see the timing doc).
- **The v1 accuracy analysis stands as written.** v2's own contribution is complete timing, the looped-clip pass and the long-clip pass under the pinned versions.

## 3. Accuracy by model

| Model | Params | Clean WER | All 22 corruptions | Capped WER | CER (all) | Median row WER | Rows that looped | Degradation (all / clean) |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 39M | 6.91 | 42.16 | 27.33 | 27.48 | 20.0 | 671 | 6.1x |
| whisper-base | 74M | 4.61 | 32.62 | 18.86 | 21.76 | 11.1 | 504 | 7.1x |
| whisper-small | 244M | 3.08 | 12.25 | 10.98 | 7.18 | 4.0 | 46 | 4.0x |
| whisper-medium | 769M | 2.56 | 7.90 | 7.69 | 4.34 | 0.0 | 6 | 3.1x |
| whisper-large-v3 | 1.55B | 1.76 | **5.16** | **5.08** | **2.74** | 0.0 | 1 | **2.9x** |
| whisper-large-v3-turbo | 809M | 1.91 | 5.68 | 5.43 | 2.98 | 0.0 | 7 | 3.0x |
| wav2vec2-large-lv60-self | 317M | **1.59** | 20.00 | 19.99 | 13.72 | 3.3 | 0 | 12.6x |

- **Capped WER** limits each row's errors to its reference length, so a transcript that loops for 500 words counts as 100% wrong, not 1,700%. It changes tiny (42 to 27) and base (33 to 19) a lot, small a little, and the large models hardly at all.
- **A row loops** when its transcript compresses more than 2.4x with zlib, the test OpenAI's Whisper uses to trigger its fallback. This catches more than v1's "insertions over 2x the reference" rule (671 against 435 rows for tiny), because it also catches loops of digits ("0000...") that the normalizer squashes into one word.
- **Robustness grows with size, with one break**: base degrades more than tiny (7.1x against 6.1x), because insertions from loops are half of its errors. With capped WER the curve is flat from tiny to base (4.0x, 4.1x), then falls steadily: small 3.6x, medium 3.0x, large-v3 2.9x, turbo 2.8x.
- **Turbo is within 0.5 points of large-v3 on 19 of 22 corruptions.** The exceptions are white noise (1.0 points), `BL_500_1000` (1.6) and the combined chain (4.4).
- **wav2vec2 is best on clean audio but degrades 12.6x**, twice as much as any Whisper model.

### Error types (share of all word errors)

| Model | Substitutions | Deletions | Insertions |
|---|---|---|---|
| tiny | 43.9% | 12.4% | 43.6% |
| base | 43.7% | 6.6% | 49.7% |
| small | 67.3% | 11.1% | 21.6% |
| medium | 72.7% | 13.7% | 13.5% |
| large-v3 | 73.3% | 14.2% | 12.5% |
| turbo | 72.4% | 12.7% | 14.9% |
| wav2vec2 | 49.0% | 49.1% | 1.8% |

Whisper models fail by inventing text: insertions, which is what a loop is, and substitutions. wav2vec2 fails by dropping words: half its errors are deletions, and it almost never inserts. That matches the two designs. Whisper's decoder is a language model that keeps writing. wav2vec2's CTC head outputs nothing where it hears nothing.

## 4. Accuracy by corruption

Corpus WER %, sorted by large-v3:

| Corruption | tiny | base | small | medium | large-v3 | turbo | wav2vec2 |
|---|---|---|---|---|---|---|---|
| none | 6.9 | 4.6 | 3.1 | 2.6 | 1.8 | 1.9 | **1.6** |
| codec_8k | 9.4 | 5.9 | 3.7 | 2.7 | 1.9 | 2.0 | 1.9 |
| PL_40_.1 | 9.6 | 6.1 | 3.5 | 2.5 | 1.9 | 2.0 | **1.8** |
| clip_1 | 14.7 | 9.0 | 4.5 | 3.3 | 2.0 | 2.4 | 2.7 |
| clipN_1 | 14.9 | 8.9 | 4.4 | 3.2 | 2.0 | 2.3 | 2.7 |
| SV_0.5 | 12.0 | 7.0 | 4.2 | 2.9 | 2.1 | 2.2 | 2.2 |
| codec_6k | 11.3 | 7.9 | 4.5 | 3.1 | 2.2 | 2.4 | 2.3 |
| PL_30_.2 | 15.9 | 9.4 | 5.0 | 3.5 | 2.2 | 2.5 | 2.6 |
| PL_20_.3 | 20.4 | 12.3 | 5.8 | 3.9 | 2.5 | 2.6 | 3.1 |
| BL_800_3000 | 32.1 | 15.3 | 5.7 | 3.9 | 2.5 | 2.6 | 3.7 |
| PL_10_.4 | 22.8 | 13.1 | 5.7 | 3.7 | 2.6 | 2.8 | 10.4 |
| clipN_05 | 21.5 | 13.3 | 6.2 | 4.3 | 2.6 | 2.9 | 8.5 |
| clip_05 | 22.5 | 14.3 | 6.2 | 4.3 | 2.6 | 2.9 | 8.5 |
| PL_5_.5 | 26.4 | 16.1 | 6.9 | 4.3 | 2.7 | 3.2 | 83.1 |
| SDO_8_100_15 | 17.2 | 12.6 | 7.3 | 5.4 | 3.2 | 3.6 | 3.4 |
| BL_600_2000 | 74.1 | 28.1 | 9.9 | 5.5 | 3.2 | 3.4 | 9.5 |
| RV_0.5_251 | 26.2 | 15.5 | 7.3 | 4.6 | 3.3 | 3.4 | 8.6 |
| clipN_025 | 34.2 | 22.1 | 9.9 | 5.8 | 3.4 | 3.7 | 34.4 |
| clip_025 | 33.9 | 22.7 | 9.2 | 5.9 | 3.5 | 3.8 | 34.4 |
| WN_3snr | 37.2 | 24.3 | 12.7 | 8.1 | 5.5 | 6.5 | 39.7 |
| WN_3-SV_.5-SDO_8_100_15 | 127.1 | 103.4 | 47.0 | 33.7 | 26.3 | 30.7 | 86.0 |
| BL_500_1000 | 337.4 | 345.8 | 96.8 | 56.7 | 33.4 | 35.0 | 88.9 |

Capped WER for the corruptions where capping matters (the others change by less than 2 points):

| Corruption | tiny | base | small | medium | large-v3 | turbo |
|---|---|---|---|---|---|---|
| BL_500_1000 | 98.3 | 95.8 | 75.6 | 52.6 | 33.0 | 34.8 |
| WN_3-SV_.5-SDO_8_100_15 | 77.9 | 66.0 | 43.2 | 33.5 | 25.1 | 26.0 |
| BL_600_2000 | 57.5 | 26.7 | 9.1 | 5.5 | 3.2 | 3.4 |
| clip_025 | 30.2 | 20.2 | 9.2 | 5.9 | 3.5 | 3.8 |
| clipN_025 | 29.5 | 19.5 | 9.1 | 5.8 | 3.4 | 3.7 |

Observations:
- **large-v3 is the best model on 20 of 22 corruptions.** wav2vec2 wins the other two, clean audio and light packet loss (`PL_40_.1`), by 0.1-0.2 points.
- **Most corruptions barely touch the large models**: 19 of 22 leave large-v3 at 3.5% or below. Only white noise (5.5%), the 500-1000 Hz band (33%) and the combined noise, reverb and dropout chain (26%) hurt it.
- **Band limiting is the hardest family.** At 800-3000 Hz (roughly a phone line), small and up are fine. At 500-1000 Hz, which keeps almost no consonant information, every model fails, and tiny and base fall into loops on more than a quarter of the clips.
- **Clipping**: normalised (`clipN_*`) and plain (`clip_*`) clipping give nearly the same WER at each threshold: within 1 point for tiny and base and within 0.7 for the rest. Peak normalisation before clipping barely matters.
- **wav2vec2 has three cliffs** where Whisper has none:
  - `PL_5_.5` (half of all 5 ms packets dropped): 83% against 2.7% for large-v3.
  - heavy clipping (`clip_025`): 34%.
  - white noise at 3 dB SNR: 40%.

  Whisper's decoder fills in short gaps from context, and a CTC model can't.

## 5. Repetition loops

Loop rows (zlib ratio > 2.4) by corruption, Whisper models only:

| Corruption | tiny | base | small | medium | large-v3 | turbo |
|---|---|---|---|---|---|---|
| BL_500_1000 | 542 | 431 | 37 | 5 | 0 | 0 |
| WN_3-SV_.5-SDO_8_100_15 | 67 | 45 | 5 | 0 | 1 | 5 |
| BL_600_2000 | 25 | 3 | 1 | 0 | 0 | 0 |
| clipN_025 + clip_025 | 16 | 10 | 1 | 0 | 0 | 0 |
| 14 other corruptions | 21 | 15 | 2 | 1 | 0 | 2 |
| **Total** | **671** | **504** | **46** | **6** | **1** | **7** |

- **92% of loops are on two corruptions**: the 500-1000 Hz band and the combined chain. On `BL_500_1000`, 27% of tiny's clips and 22% of base's loop.
- **Loops almost vanish from medium up** (6, 1 and 7 rows out of 44,000 each). Turbo loops once on clean audio.
- **Loops are expensive as well as wrong**: they cost 15% of all GPU time in v2 (see the timing doc).

## 6. The looped-clip pass: what temperature fallback does

After the main run, the 1,077 clips (959 samples) where any Whisper model looped were rerun on all 6 Whisper models with OpenAI-style temperature fallback (`WHISPER_FALLBACK=True`). The corrupted audio is the same as in the main run, so the two transcripts compare directly. 6,462 rows.

On the rows that looped in the main run:

| Model | Looped rows | WER greedy | WER fallback | Capped WER fallback | Still looping | Fallback better / worse |
|---|---|---|---|---|---|---|
| tiny | 671 | 956 | 98 | 92.5 | 23 | 522 / 43 |
| base | 504 | 893 | 106 | 94.1 | 14 | 456 / 37 |
| small | 46 | 957 | 81 | 77.3 | 0 | 39 / 3 |
| medium | 6 | 1086 | 82 | 76.0 | 0 | 6 / 0 |
| large-v3 | 1 | 6329 | 171 | 100.0 | 0 | 1 / 0 |
| turbo | 7 | 2644 | 67 | 63.4 | 0 | 7 / 0 |

On all 1,077 rerun clips, including the ones a model didn't loop on:

| Model | WER greedy | WER fallback | Capped greedy | Capped fallback | Same transcript | Better / worse |
|---|---|---|---|---|---|---|
| tiny | 585.3 | 98.0 | 97.5 | 93.3 | 134 | 596 / 126 |
| base | 541.6 | 98.6 | 91.7 | 88.6 | 214 | 613 / 178 |
| small | 108.2 | 69.2 | 68.8 | 67.6 | 744 | 143 / 115 |
| medium | 52.5 | 47.2 | 46.8 | 46.7 | 954 | 49 / 49 |
| large-v3 | 31.4 | 29.7 | 29.5 | 29.5 | 1,041 | 14 / 12 |
| turbo | 38.2 | 30.3 | 30.4 | 30.2 | 1,023 | 22 / 15 |

Whole-run WER if the fallback transcripts replace the greedy ones on those clips:

| | tiny | base | small | medium | large-v3 | turbo |
|---|---|---|---|---|---|---|
| Greedy (main run) | 42.16 | 32.62 | 12.25 | 7.90 | 5.16 | 5.68 |
| With fallback on looped clips | 27.59 | 19.38 | 11.08 | 7.74 | 5.11 | 5.44 |
| Capped greedy | 27.33 | 18.86 | 10.98 | 7.69 | 5.08 | 5.43 |

What this shows:
- **The fallback breaks 96-100% of loops**: tiny 671 to 23, base 504 to 14, the rest to 0.
- **It stops the runaway, but it doesn't recover the speech.** On looped rows, WER drops from about 900% to about 100%, and capped WER only from about 100% to 92-94% for tiny and base. The audio is unintelligible to these models. The fallback swaps an endless loop for a short wrong guess.
- **So capped WER is a good stand-in for "Whisper with its usual safety net".** The whole-run WER with fallback rows substituted is within 0.5 points of capped greedy WER for every model. Reporting capped WER next to corpus WER gives the realistic number without running fallback on all 308,000 clips.
- **On clips that didn't loop, the fallback is neutral.** medium, large-v3 and turbo keep 89-97% of transcripts identical, and better and worse changes roughly balance. That's because a low average log-probability also triggers resampling, which can go either way.
- 8 of tiny's fallback transcripts are empty: the model gave up instead of looping.

## 7. Clips over 30 s

8 samples have a clip over 30 s. After reverberation, all 22 corruptions of each of them are over 30 s, so this is 176 clips per model. The main run decoded them in 30 s chunks; the long-clip pass reran them with OpenAI's sequential long-form decoding.

| Model | Chunked (main run) | Sequential (v2) | Sequential (v1) |
|---|---|---|---|
| tiny | **28.6** | 39.0 | 39.0 |
| base | 24.0 | **22.4** | 22.4 |
| small | 18.0 | **12.1** | 12.1 |
| medium | 9.7 | **8.0** | 8.0 |
| large-v3 | 6.8 | **5.9** | 5.9 |
| turbo | 6.3 | **5.7** | 5.7 |

- **Same as v1**: sequential decoding is better for base and up (by 6 points on small), and worse for tiny, which hallucinates more with timestamps.
- **Long clips are harder for every model**: capped WER on clips over 30 s is 1.2x (tiny) to 1.9x (large-v3: 6.8% against 3.5%) the 20-30 s figure. Part of that is chunking, and part is that these 8 samples are just 8 speakers. Treat it as indicative.

## 8. Clip length and audio-health flags

Capped WER by clip duration, all corruptions:

| Duration | Rows per model | tiny | base | small | medium | large-v3 | turbo | wav2vec2 |
|---|---|---|---|---|---|---|---|---|
| 0-5 s | 17,732 | 33.4 | 24.6 | 15.5 | 11.8 | 8.3 | 8.6 | 20.4 |
| 5-10 s | 15,158 | 28.1 | 19.5 | 11.1 | 7.9 | 5.4 | 5.7 | 19.4 |
| 10-15 s | 6,512 | 25.6 | 17.1 | 9.4 | 6.0 | 3.7 | 4.2 | 19.5 |
| 15-20 s | 2,838 | 24.0 | 15.5 | 8.9 | 5.7 | 3.5 | 3.8 | 20.2 |
| 20-30 s | 1,584 | 22.4 | 15.2 | 8.5 | 5.6 | 3.5 | 3.9 | 21.5 |
| 30-40 s | 176 | 25.8 | 19.3 | 13.1 | 9.7 | 6.8 | 6.3 | 25.3 |

- **Short clips are hardest for Whisper**: large-v3 is 2.4x worse on clips under 5 s than on 15-30 s clips. A short clip gives the decoder little context, and every error weighs more against fewer words.
- **wav2vec2 is flat across lengths** (19-22%). It decodes each frame on its own, so context length doesn't help it.

The pipeline's audio-health check flags 2,205 of 44,000 clips as needing action, mostly hard clipping (`clip_025`: 1,123) and white noise (460). The flag doesn't predict Whisper errors: large-v3 does slightly better on flagged clips (4.6% against 5.1%). It does predict wav2vec2 errors (26.8% against 19.7%). The flag picks up clipping and noise, which hurt wav2vec2. The band limits and dropouts that hurt Whisper go unflagged.
