#!/usr/bin/env python3
# layer_b/broker_client.py
"""Shared MQTT bus wrapper used by Layer B modules."""
import paho.mqtt.client as mqtt
import json


class Bus:
  def __init__(self, host="localhost", port=1883):
    self._topics = []   # every topic ever subscribed, for re-subscribe on reconnect
    self.client = mqtt.Client()
    # paho auto-reconnects after a broker restart, but the default clean
    # session means the broker forgets our subscriptions - without
    # re-subscribing here every module would come back connected but
    # silently deaf to everything.
    self.client.on_connect = self._on_connect
    self.client.connect(host, port, 60)
    self.client.loop_start()

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