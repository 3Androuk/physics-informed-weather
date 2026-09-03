#!/bin/bash
# usage: run_fcst_sm.sh <eta> <lead>  -- combo with saved members for inflation
B=/cs/student/project_msc/2025/ml/ahakim/physics-informed-weather
cd $B/flow-stochastic-superres/era5-diffusion-downscaling
ETA=$1; LEAD=$2; TAG=combo_eta${ETA}_${LEAD}
echo "########## combo eta=$ETA lead ${LEAD}h (save members) ##########"
$B/.venv/bin/python -u -m eval.downscale_forecast --config config/t2m.yaml \
    --ckpt diffusion_geo_combo.pt --data-dir datasets/forecast_hres_t2m \
    --lead $LEAD --limit 8 --ensemble 4 --control --tile 144 --eta $ETA \
    --save-members results_t2m/members/combo_eta${ETA}
echo "FC_${TAG}_DONE rc=$?"
$B/.venv/bin/python -u -m eval.inflate_members --members results_t2m/members/combo_eta${ETA} \
    --data-dir datasets/forecast_hres_t2m --lead $LEAD
echo "FCST_${TAG}_ALLDONE"
