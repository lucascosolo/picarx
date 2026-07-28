import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402,F401

from camera_client import CameraSubscription  # noqa: E402
from camera_controller import SubscriptionBook  # noqa: E402


class CameraControllerTests(unittest.TestCase):
    def test_book_expires_subscriptions_and_uses_highest_requested_rate(self):
        now = [100.0]
        book = SubscriptionBook(clock=lambda: now[0])
        book.update({"subscriber": "vision", "fps": 4, "ttl": 2})
        book.update({"subscriber": "gesture", "fps": 10, "ttl": 2})
        self.assertEqual(book.max_fps(), 10.0)
        now[0] = 102.1
        self.assertEqual(book.active(), [])
        self.assertEqual(book.max_fps(), 0.0)

    def test_update_is_idempotent_and_disable_releases_one_subscriber(self):
        book = SubscriptionBook(clock=lambda: 100.0)
        payload = {"subscriber": "vision", "enabled": True, "fps": 4}
        book.update(payload)
        book.update(payload)
        self.assertEqual(len(book.active()), 1)
        book.update({"subscriber": "vision", "enabled": False})
        self.assertEqual(book.active(), [])

    def test_client_refreshes_a_short_lived_subscription(self):
        bus = harness.FakeBus()
        subscription = CameraSubscription(bus, "gesture", 10)
        subscription.ensure(now=0.0)
        self.assertEqual(bus.last("picarx/camera/subscribe")["fps"], 10.0)
        subscription.ensure(now=0.5)
        self.assertEqual(len(bus.of("picarx/camera/subscribe")), 1)
        subscription.ensure(now=0.8)
        self.assertEqual(len(bus.of("picarx/camera/subscribe")), 2)
        subscription.release()
        self.assertFalse(subscription.active)
        self.assertFalse(bus.last("picarx/camera/subscribe")["enabled"])


if __name__ == "__main__":
    unittest.main()
