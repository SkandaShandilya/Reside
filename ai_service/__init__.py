# AI Service package
"""
AI Service for Suspicious Activity Detection

Modules:
- detector.py: YOLO person/object detection
- tracker.py: Object tracking (DeepSORT/ByteTrack)
- activity_model.py: Activity recognition (Time-Distributed CNN / Conv3D)
- behaviour_engine.py: Rule-based + ML behaviour analysis
- video_processor.py: Frame extraction and preprocessing
- alert_generator.py: Alert creation with deduplication
- inference_engine.py: Real-time inference orchestrator
- zone_detector.py: Restricted zone analysis

Training Pipeline (training/):
- dataset.py: Dataset loading and augmentation
- train.py: Training loop
- evaluate.py: Evaluation metrics
- fine_tune.py: Fine-tuning on verified samples
"""
