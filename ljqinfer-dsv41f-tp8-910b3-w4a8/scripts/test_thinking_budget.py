"""CPU policy tests; no weights or NPU required."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.thinking import ThinkingBudget, end_bias, validate_budget, THINK_START, THINK_END, ASSISTANT

class PolicyTests(unittest.TestCase):
    def test_curve(self):
        for n in (0,1,899,900): self.assertEqual(end_bias(n,1000),0)
        for n,b in ((950,8),(980,20.48),(1000,32),(2000,32)):
            self.assertAlmostEqual(end_bias(n,1000),b)
        self.assertEqual(end_bias(100,0),0)
        self.assertEqual(end_bias(1,1),32)

    def test_request_scope(self):
        s=ThinkingBudget.from_prompt([ASSISTANT,THINK_START,5,THINK_END,128803,4,ASSISTANT,THINK_START],100)
        self.assertTrue(s.active);self.assertEqual(s.count,0)
        s=ThinkingBudget.from_prompt([ASSISTANT,THINK_START,5,THINK_END,128803,4,ASSISTANT,THINK_END],100)
        self.assertFalse(s.active)
        self.assertFalse(ThinkingBudget.from_prompt([THINK_START,128803,5],100).active)

    def test_speculative_close_is_not_committed(self):
        s=ThinkingBudget(100,True,95)
        b=s.preview([123,456,THINK_END,7,8,9])
        self.assertAlmostEqual(b[0],8);self.assertGreater(b[1],b[0]);self.assertEqual(b[2:],[0]*4)
        self.assertTrue(s.active);self.assertEqual(s.count,95)
        s.accept([456,789]) # close was rejected
        self.assertTrue(s.active);self.assertEqual(s.count,97)
        s.accept([THINK_END,77]);self.assertFalse(s.active)
        self.assertEqual(s.preview([77,4,5,6,7,8]),[0]*6)

    def test_rebatch_and_fresh_slot(self):
        a,b=ThinkingBudget(100,True,99),ThinkingBudget(200,True,99)
        self.assertGreater(a.preview([1]*6)[0],0)
        self.assertEqual(b.preview([1]*6),[0]*6)
        self.assertEqual(ThinkingBudget().preview([1]*6),[0]*6)
        a.accept([THINK_END]);self.assertEqual(a.preview([1]*6),[0]*6)

    def test_guard(self):
        for x in (True,1.2,'12',-1,2**31):
            with self.assertRaises(ValueError): validate_budget(x)
        self.assertEqual(validate_budget(None),0)
        self.assertEqual(validate_budget(4096),4096)

class ServiceBudgetTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        from server.service import ServiceLayer
        for request in ({'thinking': True}, {'thinking': {'type': 'enabled'}},
                        {'reasoning_effort': 'high'}):
            for limit in (1, 2, 101, 1000):
                plan = ServiceLayer._thinking_options(request, limit)
                self.assertTrue(plan['enable_thinking'])
                self.assertEqual(plan['thinking_budget_tokens'], limit * 9 // 10)
        plan = ServiceLayer._thinking_options(
            {'thinking': {'type': 'enabled', 'budget_tokens': 123}}, 1000)
        self.assertEqual(plan['thinking_budget_tokens'], 123)
        for request in ({}, {'thinking': False}, {'thinking': {'budget_tokens': 0}},
                        {'thinking': {'type': 'disabled'}, 'reasoning_effort': 'high'},
                        {'thinking': True, 'reasoning_effort': 'none'}):
            plan = ServiceLayer._thinking_options(request, 1000)
            self.assertFalse(plan['enable_thinking'])
            self.assertIsNone(plan['thinking_budget_tokens'])

    def test_effective_output_limit(self):
        from server.service import ServiceLayer, ServiceError
        service = ServiceLayer.__new__(ServiceLayer)
        service.max_output_tokens = 100
        service.model_name = 'test'
        request = {'messages': [{'role': 'user', 'content': 'hello'}],
                   'max_tokens': 1000, 'thinking': True}
        plan = service.intake(request)
        self.assertEqual(plan['max_tokens'], 100)
        self.assertEqual(plan['thinking_budget_tokens'], 90)
        request['thinking'] = {'type': 'enabled', 'budget_tokens': 101}
        with self.assertRaises(ServiceError):
            service.intake(request)

if __name__=='__main__': unittest.main()
