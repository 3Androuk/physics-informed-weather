#!/bin/bash
# Regenerate docs/forecast_demo_results.md and push it if it changed.
B=/cs/student/project_msc/2025/ml/ahakim/physics-informed-weather
cd $B/flow-stochastic-superres/era5-diffusion-downscaling
$B/.venv/bin/python -m eval.forecast_results_md || exit 1
if ! git diff --quiet -- docs/forecast_demo_results.md; then
  git add docs/forecast_demo_results.md
  git commit -q -m "docs: forecast demo results table (auto-regenerated)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
  git push -q origin HEAD && echo "pushed updated results md" || echo "PUSH FAILED"
else
  echo "no change"
fi
