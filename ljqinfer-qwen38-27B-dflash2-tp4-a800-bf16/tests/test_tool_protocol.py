import json
import unittest
from server.service import ServiceLayer, parse_tool_calls, split_reasoning


class ToolProtocolTest(unittest.TestCase):
    def test_split_reasoning_keeps_only_visible_answer(self):
        reasoning, content = split_reasoning(
            "Need inspect result.\n</think>\n\n北京现在晴，26°C。")
        self.assertEqual(reasoning, "Need inspect result.")
        self.assertEqual(content, "北京现在晴，26°C。")

    def test_split_reasoning_without_marker_is_visible(self):
        self.assertEqual(split_reasoning("plain answer"), ("", "plain answer"))

    def test_parse_qwen_xml(self):
        text = ('reasoning</think>\n<tool_call>\n<function=get_weather>\n'
                '<parameter=city>\n北京\n</parameter>\n'
                '<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>')
        content, calls = parse_tool_calls(text)
        self.assertEqual(content, 'reasoning</think>')
        self.assertEqual(calls[0]['function']['name'], 'get_weather')
        self.assertEqual(json.loads(calls[0]['function']['arguments']),
                         {'city': '北京', 'days': 2})

    def test_openai_argument_string_is_mapping_for_template(self):
        messages = [{'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'call_1', 'type': 'function', 'function': {
                'name': 'get_weather', 'arguments': '{"city":"北京"}'}}]}]
        normalized = ServiceLayer.__new__(ServiceLayer)._template_messages(messages)
        self.assertEqual(normalized[0]['tool_calls'][0]['function']['arguments'],
                         {'city': '北京'})
        self.assertIsInstance(messages[0]['tool_calls'][0]['function']['arguments'], str)

    def test_invalid_argument_string_is_rejected(self):
        messages = [{'role': 'assistant', 'tool_calls': [{
            'function': {'name': 'bad', 'arguments': '{'}}]}]
        with self.assertRaisesRegex(ValueError, 'valid JSON'):
            ServiceLayer.__new__(ServiceLayer)._template_messages(messages)


if __name__ == '__main__':
    unittest.main()
