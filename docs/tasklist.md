# TransBloom Feature Tasklist

Use this file as the execution checklist. Mark each item as complete by changing `[ ]` to `[x]`.

## 0. Current State Snapshot (as of 2026-03-10)

### 0.1 Implemented Baseline (Already Built)
- [x] Prepare binary `flower` / `non_flower` dataset from Oxford Flowers102 + CIFAR-100.
- [x] Train FlowerNet and save checkpoints (`best_model.pt`, `last_model.pt`) plus `class_to_idx.json`.
- [x] Evaluate checkpoints with confusion matrix + per-class metrics and optional JSON report output.
- [x] Export trained model to ONNX (`flowernet.onnx`) with documented input/output contract.
- [x] Run local ONNX inference for single-image predictions and JSON output.
- [x] Ship Android app with CameraX preview, still-frame detection, ONNX Runtime inference, and overlayed prediction/confidence.
- [x] Bundle model and class map assets in the Android app (`flowernet.onnx`, `class_to_idx.json`).

### 0.2 Immediate Gaps Relative to Production
- [ ] Add automated regression tests/CI for data, training, export, inference, and Android.
- [ ] Add cascade policy, backend API fallback, and HITL/review pipeline.

## 1. Baseline Stabilization (CNN + Mobile)

### 1.1 Model and Training Hardening
- [x] Emit train/val metrics to console during training runs (per epoch).
- [ ] Add structured run metadata output for every training run (`model_version`, `dataset_version`, `seed`, metrics artifact).
- [ ] Add deterministic training option (`--seed`, deterministic flags) for reproducible debugging.
- [ ] Add confidence calibration step (temperature scaling) and save calibration parameters.
- [ ] Define and document production thresholds (`flower_positive_threshold`, `fallback_threshold`).

### 1.2 Data Pipeline Hardening
- [ ] Add dataset manifest generation (`counts`, class balance, source breakdown, hashes).
- [x] Validate split directory existence before training (`data/train`, `data/val`).
- [x] Enforce binary class count (`2`) before training/eval runs.
- [ ] Validate required class names (`flower`, `non_flower`) and minimum image counts per split.
- [ ] Add duplicate detection pass (exact hash) across train/val to prevent leakage.
- [ ] Add script to snapshot dataset version (`dataset_vYYYYMMDD_x` metadata record).

### 1.3 Mobile App Readiness
- [x] Load ONNX/class-map assets at app startup and fail visibly on init error.
- [ ] Show model/class-map version in app UI/debug overlay.
- [ ] Add remote-config-ready threshold fields in app (without hardcoding only in code).
- [ ] Add app-side fallback queue model (local persistent queue for uncertain cases).
- [ ] Add connectivity state handling (retry policy, backoff, max queue size).

## 2. Cascade Decision Engine (CNN On-Device + Transformer API Fallback)

### 2.1 Decision Rules
- [ ] Implement policy `if CNN predicts flower and confidence < 0.75 -> call API`.
- [ ] Implement policy `if CNN predicts non_flower and confidence < secondary threshold -> call API`.
- [ ] Implement policy `if CNN and Transformer disagree -> mark for mandatory review`.
- [ ] Add policy versioning (`decision_policy_version`) in outputs and logs.

### 2.2 API Contract and Service
- [ ] Define OpenAPI spec for `POST /flowernet/flowers`.
- [ ] Define request schema: image payload, cnn label/confidence, model versions, device metadata.
- [ ] Define response schema: transformer label/confidence, final decision, trace id, model version.
- [ ] Build inference API skeleton with auth, request validation, and structured logging.
- [ ] Add object storage upload path for incoming fallback images.
- [ ] Add DB table for inference events and review status.

### 2.3 Observability
- [ ] Add metrics: fallback rate, disagreement rate, avg inference latency, API error rate.
- [ ] Add alert thresholds for API failures and queue backlog growth.
- [ ] Add daily quality dashboard rollup (precision/recall from labeled subset).

## 3. Human-in-the-Loop (HITL) Feedback System

### 3.1 Feedback Capture
- [ ] Add in-app actions: `Correct`, `Wrong -> Flower`, `Wrong -> Non_flower`.
- [ ] Save user feedback record with image reference, original prediction, corrected label, timestamp.
- [ ] Add consent/privacy notice and data-retention settings for user-submitted feedback.

### 3.2 Review Queue
- [ ] Build review queue prioritization:
- [ ] Include disagreements.
- [ ] Include low-confidence transformer cases.
- [ ] Include random sample for drift monitoring.
- [ ] Add reviewer workflow (approve/relabel/reject).
- [ ] Record reviewer ID, decision reason, and final label.

### 3.3 Dataset Ingestion
- [ ] Add ingestion job to move approved reviewed samples into training corpus.
- [ ] Tag every ingested sample with source (`cnn_fallback`, `user_feedback`, `manual`).
- [ ] Preserve immutable audit log for all label edits.

## 4. Transformer Phase

### 4.1 Training Pipeline (Server-Side)
- [ ] Select first transformer candidate (`MobileViT` or `EfficientFormer`) and lock baseline config.
- [ ] Implement training config for transformer (`224x224`, AdamW, scheduler, augmentations).
- [ ] Add mixed-precision and checkpoint-resume support.
- [ ] Add experiment tracking for hyperparameters and outputs.

### 4.2 Evaluation and Promotion
- [ ] Build side-by-side benchmark report: CNN-only vs Transformer-only vs Cascade.
- [ ] Define promotion gates:
- [ ] Minimum precision/recall improvement.
- [ ] Maximum latency and cost per request.
- [ ] Maximum fallback/error rates.
- [ ] Add canary release process before full rollout.

### 4.3 Optional Distillation Back to Mobile
- [ ] Evaluate distilling transformer teacher into smaller mobile student model.
- [ ] Compare student accuracy/latency against current CNN.
- [ ] Decide whether to replace on-device CNN or keep as-is.

## 5. Regression Testing Tasks

### 5.1 Data and Training Regressions
- [x] `prepare_data.py` creates expected folder layout and supports balanced export mode.
- [ ] Add automated test: `prepare_data.py` creates expected folder structure and balanced counts.
- [x] Training currently writes `class_to_idx.json`.
- [ ] Add test: class-map consistency check (`flower` index and `non_flower` index stable).
- [x] Training CLI writes checkpoints (`best_model.pt`, `last_model.pt`).
- [ ] Add test: train smoke test (`--epochs 1`) completes and writes checkpoints.
- [x] Evaluate CLI outputs confusion matrix and optional JSON report.
- [ ] Add test: evaluate script runs and outputs confusion matrix + JSON report.

### 5.2 Model Export and Inference Regressions
- [x] ONNX export CLI produces `flowernet.onnx`.
- [ ] Add test: ONNX export succeeds and produces non-empty `flowernet.onnx`.
- [x] Local ONNX prediction CLI runs per-image inference and reports probabilities.
- [ ] Add test: ONNX local prediction matches PyTorch prediction on fixed sample set.
- [ ] Add test: preprocessing parity check (Python vs mobile expected tensor values).
- [ ] Add test: confidence threshold routing behavior (unit tests for decision policy).

### 5.3 Android Regressions
- [x] App runtime path loads ONNX assets and shows failure state if load fails.
- [ ] Add instrumentation test: app starts and loads ONNX assets.
- [x] Camera permission flow is implemented (request + no-permission screen path).
- [ ] Add test: camera permission flow works (grant/deny path).
- [x] Stillness detector and stability-gated inference path are implemented.
- [ ] Add test: stillness detector enters stable state on synthetic low-motion input.
- [x] Inference path emits label/confidence to overlay.
- [ ] Add test: inference path emits label/confidence without crash on sample frame.
- [ ] Add test: fallback queue stores unsent records offline and retries when network returns.

### 5.4 API and HITL Regressions
- [ ] Add API contract tests for request/response schema compatibility.
- [ ] Add test: image storage + metadata DB transaction succeeds atomically.
- [ ] Add test: disagreement cases always enter review queue.
- [ ] Add test: reviewed labels are ingested exactly once (idempotency).
- [ ] Add test: audit trail records all label updates.

### 5.5 Release Gate Checklist (Run Before Every Release)
- [ ] Data leakage check (no train/val overlap by hash).
- [ ] Full regression suite pass (data, train, export, inference, android, api).
- [ ] Accuracy gate pass on fixed holdout set.
- [ ] Latency gate pass (mobile local + API fallback).
- [ ] Rollback plan verified (previous model + policy version available).

## 6. Immediate Next Sprint (Recommended)

- [ ] Finalize decision thresholds and policy versioning.
- [ ] Implement `POST /flowernet/flowers` minimal API and logging.
- [ ] Add app fallback queue + event schema.
- [x] Core smoke-chain commands exist (`train.py`, `export_onnx.py`, `evaluate.py`, `local_predict.py`).
- [ ] Build first regression smoke workflow (train -> export -> evaluate -> local_predict) as one automated target.
- [ ] Define HITL storage schema and review queue MVP.
