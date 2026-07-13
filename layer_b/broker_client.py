#!/usr/bin/env python3
# /home/picarx/layer_b/broker_client.py
"""Shared MQTT bus wrapper used by Layer B modules."""
import paho.mqtt.client as mqtt
import json


class Bus:
  def __init__(self, host="localhost", port=1883):
    self.client = mqtt.Client()
    self.client.connect(host, port, 60)
    self.client.loop_start()

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
    """
    def _on_message(client, userdata, msg):
      callback(json.loads(msg.payload.decode()))
    self.client.message_callback_add(topic, _on_message)
    self.client.subscribe(topic)