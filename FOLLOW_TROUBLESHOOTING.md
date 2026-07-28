# Follow-mode troubleshooting

Follow mode publishes a diagnostic stream separate from its lifecycle toggle:

```bash
mosquitto_sub -v \
  -t picarx/tools/follow/status \
  -t picarx/state/current \
  -t picarx/action/result
```

The follow status `state` is one of `off`, `acquiring`, `tracking`, or
`reacquiring`. `reason` explains why it is waiting or moving. `target` is the
selected fresh source (`person` is preferred; `face` is a centering fallback),
while `sources.person` and `sources.face` show the cached observation age,
whether it is fresh, and whether the most recent detector pass actually found
that source. A cached MQTT payload with an old `observed_at` therefore cannot
look like a fresh target.

`last_intent` is the motion request follow most recently sent. `arbiter`
records the most recent resolved source and safety result, including a veto
reason when available. If another source wins, or the safety daemon vetoes a
follow request, that is visible here instead of appearing as unexplained
stillness. `robot_state` shows the latest resource owner, which is useful when
vision, gesture tracking, RC, or safety stop has control.

The production services are:

```bash
sudo journalctl -u picarx-orchestrator.service -f -o cat
sudo journalctl -u picarx-safety.service -f -o cat
sudo fuser -v /tmp/picarx-camera.lock
```

When the status is `acquiring` or `reacquiring`, follow always publishes a
stop intent and sweeps the head through the bounded pan list. It never drives
from an observation older than `FRESH_TARGET_SEC`; after `LOST_GIVEUP_SEC` it
disables itself and reports `target_lost_giveup`.
