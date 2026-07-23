#!/usr/bin/env python3
# layer_b/broker_client.py
"""Shared MQTT bus wrapper used by Layer B modules."""
import paho.mqtt.client as mqtt
import json

import heartbeat
try:
  import robot_config
except Exception:   # pragma: no cover - robot_config should always import
  robot_config = None

# Only the first Bus in a process heartbeats, so a module that happens to build
# two Bus instances doesn't emit two module heartbeats.
_HEARTBEAT_STARTED = False


class Bus:
  def __init__(self, host="localhost", port=1883):
    self._topics = []   # every topic ever subscribed, for re-subscribe on reconnect
    self._heartbeat = None
    self.client = mqtt.Client()
    # paho auto-reconnects after a broker restart, but the default clean
    # session means the broker forgets our subscriptions - without
    # re-subscribing here every module would come back connected but
    # silently deaf to everything.
    self.client.on_connect = self._on_connect
    self.client.connect(host, port, 60)
    self.client.loop_start()
    self._start_heartbeat()

  def _start_heartbeat(self):
    """Begin this process's unified module heartbeat (see heartbeat.py) unless
    disabled. Guarded so only the first Bus per process heartbeats; fail-soft so
    a heartbeat problem never stops a module coming up."""
    global _HEARTBEAT_STARTED
    if _HEARTBEAT_STARTED:
      return
    try:
      enabled = True
      interval = heartbeat.DEFAULT_INTERVAL_SEC
      if robot_config is not None:
        enabled = robot_config.get_bool(
          "observability", "heartbeat", True, env="PICARX_HEARTBEAT")
        interval = float(robot_config.get(
          "observability", "heartbeat_interval_sec",
          heartbeat.DEFAULT_INTERVAL_SEC, env="PICARX_HEARTBEAT_INTERVAL"))
      if not enabled:
        return
      _HEARTBEAT_STARTED = True
      self._heartbeat = heartbeat.start(self.publish, interval=interval)
    except Exception as e:
      print(f"Bus: heartbeat not started ({e})")

  def set_heartbeat_status(self, status_fn):
    """Optionally have this module's heartbeat carry a small self-reported
    status dict (status_fn() -> dict). No-op if the heartbeat isn't running."""
    if self._heartbeat is not None:
      self._heartbeat.status_fn = status_fn

  def _on_connect(self, client, userdata, flags, rc):
    for topic in list(self._topics):
      client.subscribe(topic)

  def publish(self, topic, payload: dict):
    self.client.publish(topic, json.dumps(payload))

  def subscribe(self, topic, callback):
    """
    Registers `callback` for messages on `topic` only.

    IMPORTANT: uses message_callback_add (a per-topic-filter callback)
    rather than setting self.client.on_message directly. on_message is
    a single global slot on the client - if subscribe() set it, then
    a module subscribing to more than one topic on the same Bus
    instance would have each subscribe() call silently overwrite the
    previous one, leaving only the last-subscribed topic's callback
    active for all incoming messages. message_callback_add avoids
    this by scoping each callback to its own topic filter, so any
    number of subscriptions on one Bus instance behave independently.

    Callbacks run on paho's single network thread; an exception thrown
    out of one (a malformed payload, a bug in any handler) would kill
    that thread and leave the whole module silently deaf, so every
    delivery is guarded here rather than trusting each module to
    never raise.
    """
    def _on_message(client, userdata, msg):
      try:
        payload = json.loads(msg.payload.decode())
      except (ValueError, UnicodeDecodeError) as e:
        print(f"Bus: dropping non-JSON payload on {msg.topic}: {e}")
        return
      try:
        callback(payload)
      except Exception as e:
        print(f"Bus: callback for {msg.topic} raised: {e!r}")
    self.client.message_callback_add(topic, _on_message)
    self._topics.append(topic)
    self.client.subscribe(topic)