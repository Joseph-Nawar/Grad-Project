"""One fresh-process Phase 4 assessment worker; stdout's final line is JSON."""
from __future__ import annotations
import json
import time
import psutil
from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.evaluation.gap_closure import _fixed_input

started = time.perf_counter_ns(); service = create_default_assessment_service(); constructed = time.perf_counter_ns()
result = service.assess(_fixed_input()); finished = time.perf_counter_ns()
print(json.dumps({"construction_ms": (constructed - started) / 1e6, "assessment_ms": (finished - constructed) / 1e6, "per_modality_ms": {name: item.duration_ns / 1e6 for name, item in result.modality_executions.items()}, "successful_fusion": result.fusion is not None, "rss_bytes_at_completion": psutil.Process().memory_info().rss}))
