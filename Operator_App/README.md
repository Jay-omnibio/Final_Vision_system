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
- Rebuild the active prototype gallery from teaching data.
- Edit common runtime settings without touching raw YAML.

Dataset import and full gallery design are Phase 2.
