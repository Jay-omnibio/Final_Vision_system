# Operator App

Local web app for Phase 1 operator workflow.

## Start

```powershell
python Operator_App\server.py --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

## Scope

- Start/stop the configured runtime mode.
- Show object-passed events and saved crops.
- Review novelty `new_k` groups.
- Assign selected event crops to `class/object` teaching labels.
- Automatically rebuild the active prototype gallery, novelty known embeddings,
  and novelty calibration after saving a label.
- Edit common runtime settings without touching raw YAML.
- Clear old event/crop history when starting a clean validation run.

Dataset import and full gallery design are Phase 2.

## Phase 1 Status

- Model Status panel shows active classifier gallery, novelty gallery, known
  embeddings, calibration, subclass count, and missing-file warnings.
- Selected-crop preview is shown before assigning labels.
- Reset to Base Model switches config back to the original novelty gallery,
  known embeddings, and calibration.
- Teaching sample management supports removing wrongly assigned samples.
- Rebuild progress/status logs are shown while gallery rebuild is running.

## Novelty Learning Loop

1. Start the operator app and run the fixed novelty runtime.
2. Novelty events save crops under `data/operator_events/crops/`.
3. Select crops from a `new_k` group and assign `class/object`.
4. Click `Save Label + Rebuild`.
5. The app updates `config.yaml` so the next novelty run uses:
   - `Models/Prototype_Classifier/galleries/active_gallery.npz`
   - `Models/Novelty_Detector/artifacts/embeddings/active_known.npz`
   - `Models/Novelty_Detector/artifacts/calibration/active_novelty_mahalanobis.npz`

After that rebuild, the taught object can return as `known` with the operator
label instead of `new_k`.
