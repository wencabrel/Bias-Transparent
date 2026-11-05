#!/usr/bin/env bash
# Run TRaIL-Odom on one Boxi sequence, inside chiyunnn/trail-odom:latest.
#
# Usage: scripts/run_seq.sh <variant> <sequence> <rep>
#   variant  : V4 (upstream defaults) | V0,V1,V2,V3,V5,V6,V7 (Phase 2 ablation, Part D.1/H)
#              | V0P (paper-equivalent V0: alpha/gamma off only, prox+zerovel left on --
#                see the V0P case below) | CAL,CALP,CALROB (Phase 4 calibration)
#              | HARD,SOFT,NUIS,OURS (Phase 5 degenerate-direction-aware LiDAR handling,
#                Part D.1/H, on V0P+CAL+ROB per the guideline's table; OURS = NUIS + Part
#                E.3 profile-likelihood sigma_amb instead of a constant) | HARD4,SOFT4,
#                NUIS4,OURS4 (same mechanisms on plain V4 instead, since CAL+ROB alone was
#                shown catastrophic here -- see run_seq.sh's *4 cases and docs/DECISIONS.md
#                2026-10-15)
#   sequence : Airfield1 | Airfield2 | BikeTunnel1 | BikeTunnel2 | Park1 | Park2
#   rep      : repetition number, e.g. 1, 2, 3
#
# Writes results/<variant>/<sequence>/rep<rep>/{config.yaml,run.log,run.json,trail_output/}.
# Assumes scripts/convert_boxi_bags.py has already produced data/Boxi_ros2_sqlite/<sequence>.
set -eo pipefail

VARIANT="$1"; SEQUENCE="$2"; REP="$3"
if [ -z "$VARIANT" ] || [ -z "$SEQUENCE" ] || [ -z "$REP" ]; then
    echo "usage: $0 <variant> <sequence> <rep>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$VARIANT" in
    V4|V0|V0P|V1|V2|V3|V5|V6|V7|CAL|CALP|CALROB|CALROBP|HARD|SOFT|NUIS|OURS|HARD4|SOFT4|NUIS4|OURS4|CAL4|RADARFIX|ROB4|PROXOBS|PROXOBS2|RCOND|PROXOBSC|PAPERN|PROXOBSN|ELEV|PROXOBSE|ELEVB|PROXOBSEB|PROXOBSEBG|PROXOBSEG|PLANERES|RADARMAP|SOLVCAP|WINCAP|PROXOBSEW|PROXOBSEGW|PROXOBSEWA|PROXOBSEGWA|PROXOBSEWF|PROXOBSEGWF|PROXOBSW|PROXOBSGW|PROXOBSGWF|PROXOBSEGF|PROXOBSEGWF_NOGAIN|PROXOBSEGWF_PRIORALWAYS|PROXOBSEGWF_PRIORNEVER|PROXOBSEGWF_L075|PROXOBSEGWF_L025|PROXOBSEGWF_HMN02|PROXOBSEGWF_HMP02|PROXOBSEGWF_W05|PROXOBSEGWF_W20|REF|PROXOBSEGWF_DOWN|PROXOBSEGWR|PROXOBSEGWF_L070|P2RAW*|P2COR*|P3BASE*|P3BIAS*) BASE_CFG="$REPO_ROOT/configs/base/boxi.yaml" ;;
    *) echo "unknown variant '$VARIANT' (expected one of V0,V0P,V1,V2,V3,V4,V5,V6,V7,CAL,CALP,CALROB,CALROBP,HARD,SOFT,NUIS,OURS,HARD4,SOFT4,NUIS4,OURS4,CAL4,RADARFIX,ROB4,PROXOBS,PROXOBS2,RCOND,PROXOBSC,PAPERN,PROXOBSN,ELEV,PROXOBSE,ELEVB,PROXOBSEB,PROXOBSEBG,PROXOBSEG,PLANERES,RADARMAP,SOLVCAP,WINCAP,PROXOBSEW,PROXOBSEGW,PROXOBSEWA,PROXOBSEGWA,PROXOBSEWF,PROXOBSEGWF,PROXOBSW,PROXOBSGW,PROXOBSGWF,PROXOBSEGF,PROXOBSEGWF_NOGAIN,PROXOBSEGWF_PRIORALWAYS,PROXOBSEGWF_PRIORNEVER,PROXOBSEGWF_L075,PROXOBSEGWF_L025,PROXOBSEGWF_HMN02,PROXOBSEGWF_HMP02,PROXOBSEGWF_W05,PROXOBSEGWF_W20,REF,PROXOBSEGWF_DOWN,PROXOBSEGWR,PROXOBSEGWF_L070,P2RAW<k>,P2COR<k>,P3BASE<k>,P3BIAS<k>)" >&2; exit 1 ;;
esac

# Phase 2 ablation mapping (Part D.1/H Phase 2). V4 (upstream) leaves every
# TRAIL_RESEARCH_ABLATION_* variable unset, so AblationConfig's defaults (all on,
# kappa_thresh=5, eta=2.2) apply -- i.e. V4 is exactly upstream behaviour, unchanged.
declare -a ABLATION_ENV=()
case "$VARIANT" in
    V4) ;;  # all defaults (upstream)
    # V4 + AblationConfig::SolverTimeCapS=10 (added 2026-10-28 investigating BikeTunnel2's stall).
    # Everything else stays at upstream defaults. NEGATIVE RESULT: did not fix the stall (see WINCAP).
    SOLVCAP) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_SOLVER_TIME_CAP_S=10) ;;
    # 2026-10-28 WINCAP: bound the incremental window (AblationConfig::MaxWindowS) so a radar gap
    # cannot make it grow without limit (BikeTunnel2). 1.0 s fixed a priori. PROXOBSEW = accepted
    # PROXOBSE + that bound. See docs/DECISIONS.md.
    WINCAP) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    PROXOBSEW) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    PROXOBSEGW) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    # 2026-10-29 adaptive elevation shrink (AblationConfig::AdaptiveElevShrink): lambda from the
    # window's own radar geometry, one global model; 0.5 remains the < 3-target fallback.
    PROXOBSEWA) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_ADAPTIVE_ELEV=1) ;;
    PROXOBSEGWA) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_ADAPTIVE_ELEV=1) ;;
    # 2026-10-29 floor-multipath gate (AblationConfig::RadarMinBodyHeight = 0.0 m, the IMU plane).
    PROXOBSEWF) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    PROXOBSEGWF) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    # 2026-11-01 paper tab:abl-inc (incremental attribution, isolated steps -- none of
    # these had been run before: every existing variant bundles PROX_OBS_GATE with
    # RADAR_ELEV_SHRINK from the first step on (PROXOBSE), so the cumulative order
    # V4 -> +window cap -> +gated prior -> +scale-free gain -> +elev shrink (=Ours-W)
    # the draft table asks for had three never-run intermediate cells.
    PROXOBSW) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
        # WINCAP + gated prior, no elev shrink, no scale-free gain. abl-inc step 2.
    PROXOBSGW) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
        # + scale-free gain, still no elev shrink. abl-inc step 3; step 4 (+elev shrink)
        # is exactly PROXOBSEGW (=Ours-W), already run.
    # 2026-11-01 paper tab:abl-loo (leave-one-out from Ours-F=PROXOBSEGWF) and tab:sens.
    PROXOBSGWF) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
        # Ours-F minus RadarElevShrink (lambda=1, i.e. no elevation correction). Fills
        # tab:abl-loo's "-regression-calibrated Doppler" row and tab:sens's lambda=1.0 cell.
    PROXOBSEGF) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
        # Ours-F minus MaxWindowS (no cap at all). Fills tab:abl-loo's "-window cap" row
        # and tab:sens's w_max=off cell.
    PROXOBSEGWF_NOGAIN) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0 TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0) ;;
        # Ours-F with EnableAlphaReweight and EnableGammaReweight both off (state_manager.cpp
        # :1769,1831): radar_weight never leaves its 1.0 init, i.e. kappa_i=1 everywhere,
        # with the measurement-level corrections (elev shrink, floor gate, window cap) still
        # applied. Fills tab:abl-loo's "-gain altogether" row.
    PROXOBSEGWF_PRIORALWAYS) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
        # Ours-F without FixRadarProxObservabilityGate: EnableRadarProx stays at its upstream
        # default (true), so the prior fires whenever degenerate_dim>0 regardless of whether
        # radar already observes the weak direction (state_manager.cpp:1919-1920) -- "always
        # fire". Fills tab:abl-loo's "-gated prior (always fire)" row.
    PROXOBSEGWF_PRIORNEVER) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0 TRAIL_RESEARCH_ABLATION_PROX=0) ;;
        # Ours-F with EnableRadarProx=0: the prior never fires (state_manager.cpp:1920 gates
        # on EnableRadarProx first). Fills tab:abl-loo's "-gated prior (never fire)" row.
    PROXOBSEGWF_L075) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.75 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    PROXOBSEGWF_L025) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.25 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    # 2026-11-02 GaRLILEO-specific lambda test: radar_diag.py's GT-corrected diagnostic on the
    # GaRLILEO radar gives Deming/GT-slope ~0.7-0.9 (results/summaries/radar_diag.csv), vs 0.5 on
    # the TRaIL-Odom radar -- testing whether lambda=0.5 overcorrects on GaRLILEO and whether
    # lambda=0.7 recovers the SlopeStair/Tunnel loss relative to the released code (Sec 6.3).
    PROXOBSEGWF_L070) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.7 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    # 2026-11-03 paper P2 (gain crossover) and P3 (injected-bias pass-through), Sec 9.3.
    # Variant names carry their kappa value: P2RAW1, P2RAW3, P2RAW10, P2RAW30, P2RAW100 (raw
    # radar: lambda=1 off, no floor gate) and P2COR<k> (corrected radar: lambda=0.5, floor
    # gate), all with MAX_WINDOW_S=1.0 (needed for BikeTunnel2 to complete; not a P2 variable).
    P2RAW*) KAPPA="${VARIANT#P2RAW}"; ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_FIXED_KAPPA="$KAPPA" TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    P2COR*) KAPPA="${VARIANT#P2COR}"; ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_FIXED_KAPPA="$KAPPA" TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    # P3BASE<k>/P3BIAS<k>: same fixed kappa, GaRLILEO only; P3BIAS additionally injects a 5%
    # forward-velocity scale bias (RadarFwdBiasEps=0.05). No elevation/floor correction in
    # either -- P3 isolates the gain mechanism (Prop 1), not the measurement corrections.
    P3BASE*) KAPPA="${VARIANT#P3BASE}"; ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_FIXED_KAPPA="$KAPPA" TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    P3BIAS*) KAPPA="${VARIANT#P3BIAS}"; ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_FIXED_KAPPA="$KAPPA" TRAIL_RESEARCH_ABLATION_RADAR_FWD_BIAS_EPS=0.05 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
        # tab:sens lambda in {0.75,0.25}, all other constants at their Ours-F values.
    PROXOBSEGWF_HMN02) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=-0.2) ;;
    PROXOBSEGWF_HMP02) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.2) ;;
        # tab:sens h_min in {-0.2,+0.2} m, all other constants at their Ours-F values.
    PROXOBSEGWF_W05) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=0.5 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    PROXOBSEGWF_W20) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=2.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
        # tab:sens w_max in {0.5,2.0} s, all other constants at their Ours-F values.
    # 2026-11-02 paper tab:diag reference: fixed-weight trajectory (alpha/gamma reweighting off,
    # as V0P) for the ground-truth-free radar diagnostic (scripts/radar_diag.py). The window cap
    # is added only so BikeTunnel2 produces a trajectory; it never engages elsewhere.
    REF) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0) ;;
    # 2026-11-02 paper tab:mech, regression-calibrated Doppler (b): Ours-F with the shrink replaced
    # by AblationConfig::RadarElevDown (same lambda^2 vertical information, no mean rescaling).
    PROXOBSEGWF_DOWN) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_DOWN=1 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    # 2026-11-02 paper tab:mech, floor gate (b): Ours-W with the Cauchy robust Doppler loss
    # (CalibrationConfig::EnableRobustDopplerLoss, c=2 whitened) instead of the height gate.
    PROXOBSEGWR) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_CALIBRATION_ROB=1) ;;
    RADARFIX) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_RADAR_FILTER_FIX=1) ;;
        # V4 + AblationConfig::FixRadarFilterFallback (research_config.h, added 2026-10-16
        # investigating Airfield1's ~2x gap vs the paper). Everything else stays at upstream
        # defaults -- this isolates the radar-inlier-filter-fallback fix alone.
    PROXOBS) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1) ;;
        # Refines V5 (full RadarFactor_prox removal -- a real win on BikeTunnel1, 1.688m vs
        # V4's 2.067m, docs/DECISIONS.md 2026-10-16) instead of removing prox outright: V4 +
        # AblationConfig::FixRadarProxObservabilityGate, gating prox on radar NOT already
        # observably covering the degenerate direction (isRadarSubspaceObservable, already
        # computed in state_manager.cpp for the alpha-reweight gate, now reused for prox
        # too) rather than discarding prox everywhere degenerate_dim>0. Keeps prox doing its
        # stated job (stabilize ICP when nothing else constrains a weak direction) exactly
        # where that's true, and stops it fighting good radar information everywhere else.
    RCOND) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_RADAR_COND_GATE=1) ;;
        # V4 + AblationConfig::RadarCondGate (added 2026-10-26): radar counts as observing
        # the LiDAR-degenerate subspace only if cond(Z Z^T) <= 30 (the paper's own stated
        # bound, docs/FACTS.md F3) instead of upstream's rank check (rel_tol 1e-8). Isolates
        # this one change on top of upstream.
    PROXOBSC) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_COND_GATE=1) ;;
        # PROXOBS + RadarCondGate: the same single definition of "radar observes the
        # degenerate subspace" now drives both the alpha/gamma/zero-vel boost and PROXOBS's
        # prox gate. Added 2026-10-26.
    PAPERN) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PAPER_N=1) ;;
        # V4 + AblationConfig::PaperNormalMatrix (added 2026-10-26): degeneracy detector and
        # gamma's sphericity use the paper's unweighted plane-normal matrix N = sum n n^T
        # (paper eq. 18) instead of the code's planarity-weighted sum (FACTS.md F1).
    PROXOBSN) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_PAPER_N=1) ;;
        # PROXOBS (our method) + PaperNormalMatrix. Added 2026-10-26.
    ELEV) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5) ;;
        # V4 + AblationConfig::RadarElevShrink=0.5 (added 2026-10-26): regression calibration
        # of the TI radar's noisy elevation (measured z-velocity attenuation 0.49, both seqs).
    PROXOBSE) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5) ;;
        # PROXOBS (our method) + RadarElevShrink=0.5. Added 2026-10-26.
    ELEVB) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_EST_BIAS=1) ;;
        # ELEV + AblationConfig::EstimateImuBias (added 2026-10-26): optimize ba/bg per window
        # as the paper states; upstream holds them constant at 0.
    PROXOBSEB) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_EST_BIAS=1) ;;
        # PROXOBSE (our method) + EstimateImuBias. Added 2026-10-26.
    PROXOBSEBG) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_EST_BIAS=1 TRAIL_RESEARCH_ABLATION_EST_GRAVITY=1) ;;
        # PROXOBSEB + AblationConfig::EstimateGravity (gravity direction refined per window;
        # departs from the paper's fixed gravity). Added 2026-10-26.
    PROXOBSEG) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1) ;;
        # PROXOBSE (our method) + AblationConfig::GammaInfoEqualize: gamma =
        # sqrt(lambda_{d+1}/lambda_min) instead of 1/S(N). Added 2026-10-26.
    PROXOBS2) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_FILTER_FIX=1) ;;
        # PROXOBS (the confirmed BikeTunnel1 win, 1.419m mean vs paper's 1.532m) + RADARFIX
        # (the radar-inlier-filter-fallback fix, neutral alone on Airfield1 but never tested
        # with PROXOBS) together -- both are real, verified fixes to different parts of the
        # same radar pipeline (prox's gate vs the inlier-filter fallback), testing whether
        # they compound. Added 2026-10-16.
    PLANERES) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PLANE_RES_SCALE=1.615384615) ;;
        # V4 + AblationConfig::PlaneResThreshScale, widening KNNDataAssociation's plane-fit
        # acceptance threshold (default 0.1m) by the MEASURED sparsity ratio between
        # Airfield1 and Park1's raw LiDAR point spacing (mean nearest-neighbour distance
        # 0.084m vs 0.052m, 100-scan rosbag2 sample each) -- not a guessed constant. Added
        # 2026-10-16 investigating Airfield1's ~2x gap vs the paper.
    RADARMAP) ABLATION_ENV=(TRAIL_RESEARCH_RADAR_MAP=1) ;;
        # V4 + RadarMapConfig::Enable -- a genuinely new mechanism (not a reweighting
        # variant of the existing architecture): adds radar target 3D positions as a
        # SECOND point source into the same LiDAR point-to-plane map constraint, reusing
        # the existing local map and factor unchanged. Upstream only ever uses radar for
        # Doppler velocity, never position/mapping. See RadarMapConfig's header comment
        # (research_config.h) for the full motivation. Default weight=5 (sigma~0.2m) is a
        # reasoned, NOT yet validated, starting point -- explicitly flagged as such.
    ROB4) ;;  # all ablation defaults (upstream/V4); ROB set below via CALIBRATION_ENV
        # ROB (E.4, fact F10: upstream builds a Cauchy robust loss for the Doppler residual
        # but never applies it, passing nullptr instead) has only ever been tested paired
        # with CAL (CALROB/CALROBP/HARD/SOFT/NUIS/OURS all set both) -- that combination is
        # the one shown catastrophic (CALROBP: ATE ~13.8m on Park1, docs/DECISIONS.md
        # 2026-10-15). ROB4 isolates the robust-loss fix ALONE on plain V4, CAL off, added
        # 2026-10-16 to test whether a real, already-identified upstream bug -- not a new
        # invented mechanism -- helps Airfield1 specifically, where radar carries nearly all
        # of the accuracy (paper's own Table I: LiDAR-only 9.627m vs full method 0.183m) and
        # a robust loss on its residuals is a plausible, well-grounded lever that was never
        # cleanly isolated.
    V0) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0
                       TRAIL_RESEARCH_ABLATION_PROX=0 TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;
    V0P) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0) ;;
        # "paper-equivalent" V0: the TRaIL paper's Table III ablation toggles ONLY alpha/gamma
        # (Sec III-F/G); it never mentions RadarFactor_prox (an ICP-init Tikhonov stabilizer
        # gated on degenerate_dim>0, state_manager.cpp:1621) or the x10 zero-vel boost
        # (:1550) -- both are undocumented-in-the-paper mechanisms this project's original
        # V0 ("nothing") also disabled, over-crippling it relative to what the paper actually
        # tested as its fixed-weight baseline. See docs/DECISIONS.md 2026-10-10 entry.
    V1) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0
                       TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # prox only
    V2) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_GAMMA=0 TRAIL_RESEARCH_ABLATION_PROX=0
                       TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # alpha only
    V3) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_PROX=0
                       TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # gamma only
    V5) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX=0) ;;  # V4 without prox
    V6) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_KAPPA_THRESH=30 TRAIL_RESEARCH_ABLATION_ETA=2.0) ;;  # paper settings
    V7) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # no x10 zero-vel boost
    CAL|CALROB) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0
                               TRAIL_RESEARCH_ABLATION_PROX=0 TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;
    CALP|CALROBP) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0) ;;
        # CAL built on V0P instead of V0 -- see V0P's comment above. CALROBP additionally
        # adds ROB (E.4) on the same V0P base -- this is the missing isolation cell between
        # CALP and HARD/SOFT/NUIS (which are also V0P+CAL+ROB, plus their own
        # degeneracy-handling mechanism on top): it measures CAL+ROB's own contribution with
        # NO HARD/SOFT/NUIS mechanism, on the exact ablation base those three use, added
        # 2026-10-15 because comparing NUIS directly against V0P (not against this cell) was
        # conflating CAL+ROB's own already-established degradation (Phase 4: CAL/CALP both
        # worse than V0P on Park1) with whatever NUIS's own delta mechanism adds on top.
    HARD|SOFT|NUIS|OURS) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0) ;;
        # Part D.1: "CAL+ROB + LiDAR info [HARD: projected out of / SOFT: attenuated along /
        # NUIS: absorbed by a nuisance shift along] U_d" -- all built on V0P (alpha/gamma
        # off, same as CALP) + CAL + ROB, plus per-residual LiDAR weighting (HARD/SOFT) or a
        # per-window nuisance-shift factor (NUIS), see
        # research/degeneracy_aware_lidar_weight.h and
        # factor/lidar_point_to_plane_nuisance_factor.hpp. HARD/SOFT's own guideline labels
        # ("DR-LRIO idea" / Hatleskog's "probabilistic SNR") are substituted per
        # docs/DECISIONS.md -- HARD is Zhang et al. 2016's hard thresholding (DR-LRIO's paper
        # contains no explicit per-window projection step to re-implement), SOFT is Hatleskog
        # & Alexis 2024's core smooth-classification idea adapted to per-residual weighting
        # (not a transplant of their exact aggregated-Hessian formula). NUIS-c is the
        # guideline's Part E.3 as specified, with a constant (not profile-likelihood) sigma_amb.
    HARD4|SOFT4|NUIS4|OURS4|CAL4) ;;  # all ablation defaults (upstream/V4), see note below
        # Deviation from the Part D.1 table, added 2026-10-15. HARD/SOFT/NUIS above build the
        # mechanism on V0P+CAL+ROB because that is what the guideline's table specifies, but
        # a direct isolation (CALROBP, docs/DECISIONS.md 2026-10-15) showed CAL+ROB alone --
        # with NO degeneracy-handling mechanism at all -- already produces ATE ~13.8m on
        # Park1, in the same range as HARD/NUIS's own results and ~15-20x worse than V4's
        # ~0.8-2m (itself close to the paper's own Table III numbers). Building a mechanism
        # on top of a base already shown catastrophic on this dataset cannot demonstrate
        # whether the mechanism itself helps. HARD4/SOFT4/NUIS4 instead apply the same
        # degeneracy-handling mechanism directly on top of plain V4 (every ablation default
        # left at upstream, no CAL, no ROB) to isolate the mechanism's own effect against the
        # one base already validated to reproduce the paper.
esac

# Phase 4 calibration mapping (Part D.1/H Phase 4). CAL = V0 + online noise calibration
# (E.2); CALROB additionally fixes the robust Doppler loss (E.4, fact F10). CALP = V0P +
# online noise calibration (the paper-equivalent-baseline version of CAL). HARD/SOFT also
# get CAL+ROB per the Part D.1 table.
declare -a CALIBRATION_ENV=()
case "$VARIANT" in
    CAL|CALP|CAL4) CALIBRATION_ENV=(TRAIL_RESEARCH_CALIBRATION_CAL=1) ;;
    CALROB|CALROBP|HARD|SOFT|NUIS|OURS) CALIBRATION_ENV=(TRAIL_RESEARCH_CALIBRATION_CAL=1 TRAIL_RESEARCH_CALIBRATION_ROB=1) ;;
    ROB4) CALIBRATION_ENV=(TRAIL_RESEARCH_CALIBRATION_ROB=1) ;;  # ROB alone, no CAL
esac
# CAL4, added 2026-10-16: CAL (E.2 online noise calibration) on the PLAIN V4 base (every
# ablation default left at upstream -- alpha/gamma/prox/zerovel all ON, no ROB), not V0 or
# V0P. CAL has only ever been tested stacked on V0/V0P -- bases already shown degraded
# independent of CAL (CALROBP: CAL+ROB alone gave ATE ~13.8m on Park1, docs/DECISIONS.md
# 2026-10-15). CAL's own documented failure mode (CalibrationConfig's FreezeKappaThresh
# guard comment, research_config.h) is specifically about degenerate-window LiDAR residual
# contamination -- Airfield1 is NOT degenerate (Phase 2: insensitive to every reweighting
# ablation; the paper's own Table I shows its accuracy comes from radar's sheer presence,
# not adaptive reweighting). CAL4 tests whether properly calibrating sigma_D to this
# specific (likely low-multipath, open-field) radar environment -- rather than the fixed
# sigma_D=0.02 assumed everywhere (docs/FACTS.md F12) -- closes any of Airfield1's gap to
# the paper's 0.183m, without CAL's known degenerate-window failure mode in play.

# Phase 5 (Part D.1/H): HARD/SOFT per-residual LiDAR weighting, see
# research/degeneracy_aware_lidar_weight.h; NUIS-c's per-window nuisance shift, see
# research_config.h's NuisanceConfig. All off by default for every other variant.
declare -a ICP_ENV=()
case "$VARIANT" in
    HARD|HARD4) ICP_ENV=(TRAIL_RESEARCH_ICP_HARD=1) ;;
    SOFT|SOFT4) ICP_ENV=(TRAIL_RESEARCH_ICP_SOFT=1) ;;
    NUIS|NUIS4) ICP_ENV=(TRAIL_RESEARCH_NUIS=1) ;;
    OURS|OURS4) ICP_ENV=(TRAIL_RESEARCH_OURS=1) ;;
        # OURS = NUIS-c's mechanism + Part E.3 profile-likelihood sigma_amb (replaces the
        # hand-picked constant with a per-window, per-axis value estimated directly from
        # the LiDAR cost landscape -- see research/ambiguity_profiler.h and
        # research_config.h's NuisanceConfig::UseProfileLikelihood).
esac
for kv in "${ICP_ENV[@]}"; do
    ABLATION_ENV+=("$kv")
done
for kv in "${CALIBRATION_ENV[@]}"; do
    ABLATION_ENV+=("$kv")
done

# Determinism fixes (applied uniformly to every variant, not a Phase 2/4 ablation axis):
# single-threaded rclcpp executor removes IMU/LiDAR/Radar callback reordering as a source
# of run-to-run variance. Opt-in via env var upstream default is multi-threaded; combined
# with tbb::parallel_deterministic_reduce (compiled in, no flag) in BuildLinearSystem.
# Boxi only: every GaRLILEO run in results/explore_garlileo/ was made without it.
case "$SEQUENCE" in
    Downstair|CorriLoop|BiCorridor|SlopeStair|Tunnel|Overpass|Quad) ;;
    *) ABLATION_ENV+=(TRAIL_RESEARCH_SINGLE_THREADED_EXECUTOR=1) ;;
esac

ABLATION_EXPORTS=""
FLAGS_JSON="{"
for kv in "${ABLATION_ENV[@]}"; do
    ABLATION_EXPORTS="${ABLATION_EXPORTS}export ${kv}
"
    key="${kv%%=*}"; val="${kv#*=}"
    FLAGS_JSON="${FLAGS_JSON}\"${key}\":\"${val}\","
done
FLAGS_JSON="${FLAGS_JSON}}"
FLAGS_JSON="$(echo "$FLAGS_JSON" | sed 's/,}/}/')"

# GaRLILEO sequences use upstream's own GaRLILEO config, bag start 0 s, and write under
# results/explore_garlileo/ (where every earlier GaRLILEO run lives); Boxi is unchanged.
case "$SEQUENCE" in
    Downstair|CorriLoop|BiCorridor|SlopeStair|Tunnel|Overpass|Quad)
        BASE_CFG="$REPO_ROOT/ws/src/TRaIL-Odom/dataset/GaRLILEO/config.yaml"
        CFG_OUT_PLACEHOLDER="/tmp/trail_results/GaRLILEO"
        BAG_REL="data/GaRLILEO/$SEQUENCE"
        OUT_REL="results/explore_garlileo/$VARIANT/$SEQUENCE/rep$REP"
        BAG_START=0 ;;
    *)
        CFG_OUT_PLACEHOLDER="/tmp/trail_results/Boxi"
        BAG_REL="data/Boxi_ros2_sqlite/$SEQUENCE"
        OUT_REL="results/$VARIANT/$SEQUENCE/rep$REP"
        BAG_START=20 ;;
esac

BAG_DIR="$REPO_ROOT/$BAG_REL"
if [ ! -d "$BAG_DIR" ]; then
    echo "missing $BAG_DIR -- run scripts/convert_boxi_bags.py $SEQUENCE first" >&2
    exit 1
fi

OUT_DIR="$REPO_ROOT/$OUT_REL"
mkdir -p "$OUT_DIR"
CFG_PATH="$OUT_DIR/config.yaml"
OUT_DIR_CONTAINER="/root/project/$OUT_REL"
sed "s|OutputPath: $CFG_OUT_PLACEHOLDER|OutputPath: $OUT_DIR_CONTAINER|" "$BASE_CFG" > "$CFG_PATH"

# OURS/OURS4 diagnostic: per-window profile-likelihood sigma_amb CSV (window_end_time,
# axis, sigma_amb, unbounded) -- the only visibility into real sigma_amb values, since
# spdlog is globally set to warn level (main.cpp:27). Off for every other variant.
case "$VARIANT" in
    OURS|OURS4) ABLATION_EXPORTS="${ABLATION_EXPORTS}export TRAIL_RESEARCH_NUIS_PL_LOG_PATH=${OUT_DIR_CONTAINER}/pl_sigma_amb.csv
" ;;
esac

# Part E.1 degeneracy detector diagnostic (kappa, degenerate_dim, sphericity, eigenvalues,
# gamma, radar inlier count) -- always on, same spdlog-suppression reason as above. Cheap
# (one CSV row per window).
ABLATION_EXPORTS="${ABLATION_EXPORTS}export TRAIL_RESEARCH_SCAN_LOG_PATH=${OUT_DIR_CONTAINER}/scan_log.csv
"
# Per-window wall-clock cost (research::WindowLogger): one row per completed window, read by
# scripts/analyze_rt.py for tab:rt. Logging only.
ABLATION_EXPORTS="${ABLATION_EXPORTS}export TRAIL_RESEARCH_WINDOW_LOG_PATH=${OUT_DIR_CONTAINER}/window_log.csv
"

CONTAINER_NAME="trail_odom_run_${VARIANT}_${SEQUENCE}_${REP}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

START_TS="$(date -Is)"
set +e
# 8 GB cap: a run that overruns it is killed alone (validation then reports no poses.txt)
# instead of pushing the desktop session into systemd-oomd kills, as on 2026-10-28.
docker run --rm -i \
    --memory=8g --memory-swap=8g \
    --name "$CONTAINER_NAME" \
    --user root \
    --hostname trail-odom \
    --entrypoint /bin/bash \
    --volume="$REPO_ROOT:/root/project" \
    chiyunnn/trail-odom:latest \
    -lc '
set -eo pipefail
source /opt/ros/humble/setup.bash
source /root/project/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
'"$ABLATION_EXPORTS"'
ros2 run rmw_zenoh_cpp rmw_zenohd &
ZENOHD_PID=$!
sleep 3

timeout 900 ros2 launch /root/project/scripts/launch/trail_boxi_headless.launch.py \
    config_path:='"$OUT_DIR_CONTAINER"'/config.yaml \
    rosbag_path:=/root/project/'"$BAG_REL"' \
    start_time:='"$BAG_START"' grace_period:=120
EXIT=$?
kill $ZENOHD_PID 2>/dev/null || true
exit $EXIT
' > "$OUT_DIR/run.log" 2>&1
RUN_EXIT=$?
set -e
END_TS="$(date -Is)"

RESEARCH_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SUBMODULE_COMMIT="$(git -C "$REPO_ROOT/ws/src/TRaIL-Odom" rev-parse HEAD)"
CONFIG_SHA256="$(sha256sum "$CFG_PATH" | cut -d' ' -f1)"

# Part I validation: exit 0 from `ros2 launch` does NOT mean trail_node reached Save() --
# it can be SIGKILLed after failing to respond to SIGINT/SIGTERM (seen on BikeTunnel2: stuck
# mid-Solve() in a near-degenerate corridor segment, falling behind real-time, never caught
# up within the shutdown grace period) and `ros2 launch` still reports success. Verify the
# actual output instead of trusting the exit code.
VALIDATION="$(python3 -c "
import json, sys
poses_path = '$OUT_DIR/trail_output/poses.txt'
errors = []
try:
    with open(poses_path) as f:
        lines = [l for l in f if l.strip()]
except OSError as e:
    print(json.dumps({'ok': False, 'errors': [f'cannot open poses.txt: {e}']}))
    sys.exit()

if len(lines) == 0:
    errors.append('poses.txt is empty')
else:
    prev_t = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 8:
            errors.append(f'line {i}: expected 8 fields, got {len(parts)}')
            break
        vals = [float(x) for x in parts]
        if any(v != v or abs(v) == float('inf') for v in vals):  # NaN or Inf
            errors.append(f'line {i}: NaN/Inf value')
            break
        if prev_t is not None and vals[0] <= prev_t:
            errors.append(f'line {i}: non-monotonic timestamp')
            break
        prev_t = vals[0]
print(json.dumps({'ok': len(errors) == 0, 'errors': errors, 'n_poses': len(lines)}))
")"

python3 -c "
import json, sys
validation = json.loads('''$VALIDATION''')
exit_code = $RUN_EXIT if validation['ok'] else 1
json.dump(dict(
    variant='$VARIANT', sequence='$SEQUENCE', rep=$REP,
    research_commit='$RESEARCH_COMMIT', trail_odom_commit='$SUBMODULE_COMMIT',
    config_sha256='$CONFIG_SHA256',
    flags=json.loads('''$FLAGS_JSON'''), command=sys.argv[1],
    start_time='$START_TS', end_time='$END_TS', host='$(hostname)',
    exit_code=exit_code, launch_exit_code=$RUN_EXIT, validation=validation,
), open('$OUT_DIR/run.json', 'w'), indent=2)
" "$0 $*"

VALIDATION_OK="$(python3 -c "import json; print(json.loads('''$VALIDATION''')['ok'])")"
if [ "$RUN_EXIT" -ne 0 ]; then
    echo "run FAILED (launch exit $RUN_EXIT) -- see $OUT_DIR/run.log" >&2
    exit "$RUN_EXIT"
fi
if [ "$VALIDATION_OK" != "True" ]; then
    echo "run FAILED validation: $VALIDATION -- see $OUT_DIR/run.log" >&2
    exit 1
fi
echo "run OK -> $OUT_DIR"
