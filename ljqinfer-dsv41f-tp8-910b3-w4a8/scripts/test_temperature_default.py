"""CPU request policy regression; no model/device required."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.service import ServiceLayer, ServiceError

class TemperatureTests(unittest.TestCase):
    def test_default_and_explicit_modes(self):
        service = ServiceLayer.__new__(ServiceLayer)
        service.max_output_tokens = 100
        service.model_name = 'test'
        base = {'messages': [{'role': 'user', 'content': 'hello'}]}
        for extra, expected in (({}, 1.0), ({'temperature': None}, 1.0),
                                ({'temperature': 0}, 0.0), ({'temperature': 1}, 1.0)):
            with self.subTest(extra=extra):
                plan = service.intake(dict(base, **extra))
                self.assertEqual(plan['temperature'], expected)
                self.assertEqual(plan['temperature_requested'], expected)
        for value in (-1, 0.5, 2, 'invalid'):
            with self.subTest(value=value), self.assertRaises(ServiceError):
                service.intake(dict(base, temperature=value))

if __name__ == '__main__': unittest.main()
