"""CPU checks of the Harmony chat server with a fake sampler (no GPU, no vLLM).

    uv run python -m unittest src/experiments/harmony_rendering/server_checks.py
"""
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from role_probe_training import harmony as h
from role_probe_training.harmony.server import HarmonyChatServer
from role_probe_training.trajectories.inspect_export import attach_histories, conversation_key

from checks import TOOLS, TURN_1, TURN_2, CONFIG


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


class FakeSampler:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def __call__(self, prompt, budget, stops, seed):
        self.calls.append(dict(prompt=list(prompt), budget=budget, stops=stops, seed=seed))
        return self.turns.pop(0)


MESSAGES = [{'role': 'system', 'content': 'Solve the task.'}, {'role': 'user', 'content': 'Weather in Paris? 😀 héllo'}]


class ServerChecks(unittest.TestCase):
    def run_conversation(self, server, url, body_extra, episode=None):
        route = url + (f'/episodes/{episode}' if episode else '') + '/v1/chat/completions'
        status, first = post(route, {'model': 'm', 'messages': MESSAGES, 'tools': TOOLS, **body_extra})
        self.assertEqual(status, 200, first)
        message = first['choices'][0]['message']
        self.assertEqual(first['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(message['reasoning_content'], 'Need the tool.')
        self.assertEqual(message['content'], '**Plan** call it.')
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'get_weather')
        history = MESSAGES + [message, {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'],
                                        'content': '{"temperature": 20}'}]
        status, second = post(route, {'model': 'm', 'messages': history, 'tools': TOOLS, **body_extra})
        self.assertEqual(status, 200, second)
        self.assertEqual(second['choices'][0]['message']['content'], 'It is 20C.')
        self.assertEqual(second['choices'][0]['finish_reason'], 'stop')
        return history + [second['choices'][0]['message']]

    def test_evaluation_mode_records_exact_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = FakeSampler([TURN_1, TURN_2])
            server = HarmonyChatServer(folder, sampler, config=CONFIG, context_limit=4096, max_tokens=512, model_name='m')
            with server.serving() as url:
                final = self.run_conversation(server, url, {'seed': 7, 'max_tokens': 512, 'reasoning_effort': 'high'})
                status, models = urllib.request.urlopen(url + '/v1/models').status, None
                self.assertEqual(status, 200)
            rows = [json.loads(line) for line in (Path(folder) / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['generation_index'] for r in rows], [0, 1])
            self.assertEqual([c['seed'] for c in sampler.calls], [7, 7 + 1000003])
            # The second prompt extends the first prompt plus its completion exactly.
            self.assertEqual(rows[1]['prompt_ids'][:len(rows[0]['prompt_ids']) + len(TURN_1)], rows[0]['prompt_ids'] + TURN_1)
            # The recorded history equals the reference rendering of the whole conversation.
            turns = [{'role': 'system', 'content': 'Solve the task.'}, {'role': 'user', 'content': 'Weather in Paris? 😀 héllo'},
                     {'role': 'assistant', 'tokens': TURN_1}, {'role': 'tool', 'name': 'get_weather', 'content': '{"temperature": 20}'}]
            reference = h.completion_prompt(h.render_turns(turns, CONFIG, tools=TOOLS)) + TURN_2
            self.assertEqual(h.history(rows[1]).ids, reference)
            self.assertEqual(rows[1]['episode_id'], conversation_key(MESSAGES))
            # The exporter joins the last record onto the trajectory row.
            row = {'sample_id': 's', 'messages': final, 'generation_records': 2}
            attach_histories([row], Path(folder) / 'requests.jsonl')
            self.assertEqual(row['history']['generation_index'], 1)
            self.assertEqual(row['unlabelled_generated_tokens'], 0)
            encoded = h.encode_record(row)
            labels = encoded['labels']
            self.assertEqual(sorted(set(labels.tolist())), [-1, 0, 1, 2, 3, 4])
            self.assertTrue((Path(folder) / 'inputs.jsonl').exists() and (Path(folder) / 'parser_events.jsonl').exists())

    def test_episode_mode_validates_seed_and_route(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = FakeSampler([TURN_1, TURN_2])
            server = HarmonyChatServer(folder, sampler, config=CONFIG, context_limit=4096, max_tokens=512,
                                       model_name='m', policy_step=3, require_episodes=True)
            with server.serving() as url:
                status, _ = post(url + '/v1/chat/completions', {'model': 'm', 'messages': MESSAGES, 'seed': 5})
                self.assertEqual(status, 500)  # plain route refused; serving then stops
            server = HarmonyChatServer(folder, FakeSampler([TURN_1, TURN_2]), config=CONFIG, context_limit=4096,
                                       max_tokens=512, model_name='m', policy_step=3, require_episodes=True)
            with server.serving() as url:
                self.assertEqual(post(url + '/episode', {'id': 'ep1', 'seed': 5})[0], 200)
                status, body = post(url + '/episodes/ep1/v1/chat/completions', {'model': 'm', 'messages': MESSAGES, 'seed': 6})
                self.assertEqual(status, 500)
                self.assertIn('seed', body['error']['message'])
            server = HarmonyChatServer(folder, FakeSampler([TURN_1, TURN_2]), config=CONFIG, context_limit=4096,
                                       max_tokens=512, model_name='m', policy_step=3, require_episodes=True)
            with server.serving() as url:
                self.assertEqual(post(url + '/episode', {'id': 'ep1', 'seed': 5})[0], 200)
                final = self.run_conversation(server, url, {'seed': 5}, episode='ep1')
                self.assertEqual(final[2]['tool_calls'][0]['id'], 'ep1_1_0')
                status, body = post(url + '/episodes/ep1/v1/chat/completions',
                                    {'model': 'm', 'messages': MESSAGES + [{'role': 'assistant', 'content': 'made up'}], 'seed': 5})
                self.assertEqual(status, 500)
                self.assertIn('retained generation tokens', body['error']['message'])
                status, _ = post(url + '/episodes/ep1/v1/chat/completions', {'model': 'm', 'messages': MESSAGES, 'seed': 5})
                self.assertEqual(status, 409)
            rows = [json.loads(line) for line in (Path(folder) / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([(r['episode_id'], r['generation_index'], r['policy_step']) for r in rows], [('ep1', 0, 3), ('ep1', 1, 3)])

    def test_context_limit_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            server = HarmonyChatServer(folder, FakeSampler([]), config=CONFIG, context_limit=50, max_tokens=512, model_name='m')
            with server.serving() as url:
                status, body = post(url + '/v1/chat/completions', {'model': 'm', 'messages': MESSAGES, 'tools': TOOLS})
            self.assertEqual(status, 200)
            self.assertEqual(body['choices'][0]['message']['content'], '')
            stops = (Path(folder) / 'episode_stops.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(stops[0])['reason'], 'serving_context_limit')

    def test_text_turns_and_temperature(self):
        """A chat that continues after a final answer (the retail benchmark), at temperature 0."""
        with tempfile.TemporaryDirectory() as folder:
            server = HarmonyChatServer(folder, FakeSampler([TURN_2]), config=CONFIG, context_limit=4096, max_tokens=512,
                                       model_name='m', temperature=0)
            with server.serving() as url:
                status, body = post(url + '/v1/chat/completions', {'model': 'm', 'messages': MESSAGES, 'temperature': 1})
                self.assertEqual(status, 500)
                self.assertIn('temperature=0', body['error']['message'])
            sampler = FakeSampler([TURN_2, TURN_1, TURN_2])
            server = HarmonyChatServer(folder + '/b', sampler, config=CONFIG, context_limit=4096, max_tokens=512,
                                       model_name='m', temperature=0)
            with server.serving() as url:
                route = url + '/v1/chat/completions'
                extra = {'temperature': 0, 'seed': 3}
                status, first = post(route, {'model': 'm', 'messages': MESSAGES, 'tools': TOOLS, **extra})
                self.assertEqual(status, 200, first)
                answer = first['choices'][0]['message']
                self.assertEqual((answer['content'], answer['reasoning_content'], first['choices'][0]['finish_reason']),
                                 ('It is 20C.', 'Got it.', 'stop'))
                # The client echoes the answer as plain text (Inspect also sends the reasoning back).
                history = MESSAGES + [{'role': 'assistant', 'content': 'It is 20C.', 'reasoning': 'Got it.'},
                                      {'role': 'user', 'content': 'Thanks'}]
                altered = history[:-2] + [{'role': 'assistant', 'content': 'It is 21C.'}, history[-1]]
                status, body = post(route, {'model': 'm', 'messages': altered, 'tools': TOOLS, **extra})
                self.assertEqual(status, 500)
                self.assertIn('differs from the text', body['error']['message'])
            server = HarmonyChatServer(folder + '/c', FakeSampler([TURN_2, TURN_1, TURN_2]), config=CONFIG,
                                       context_limit=4096, max_tokens=512, model_name='m', temperature=0)
            with server.serving() as url:
                route = url + '/v1/chat/completions'
                extra = {'temperature': 0, 'seed': 3}
                status, first = post(route, {'model': 'm', 'messages': MESSAGES, 'tools': TOOLS, **extra})
                self.assertEqual(status, 200, first)
                history = MESSAGES + [{'role': 'assistant', 'content': 'It is 20C.', 'reasoning': 'Got it.'},
                                      {'role': 'user', 'content': 'Thanks'}]
                status, second = post(route, {'model': 'm', 'messages': history, 'tools': TOOLS, **extra})
                self.assertEqual(status, 200, second)
                call = second['choices'][0]['message']
                self.assertEqual(second['choices'][0]['finish_reason'], 'tool_calls')
                history += [call, {'role': 'tool', 'tool_call_id': call['tool_calls'][0]['id'], 'content': '{"temperature": 20}'}]
                status, third = post(route, {'model': 'm', 'messages': history, 'tools': TOOLS, **extra})
                self.assertEqual(status, 200, third)
                history.append(third['choices'][0]['message'])
            rows = [json.loads(line) for line in (Path(folder) / 'c' / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['generation_index'] for r in rows], [0, 1, 2])
            self.assertEqual([r['temperature'] for r in rows], [0, 0, 0])
            # The second prompt extends the first prompt and its answer, whose <|return|> became <|end|>.
            replayed = TURN_2[:-1] + [h.tokens().end]
            self.assertEqual(rows[1]['prompt_ids'][:len(rows[0]['prompt_ids']) + len(TURN_2)], rows[0]['prompt_ids'] + replayed)
            turns = [{'role': 'system', 'content': 'Solve the task.'}, {'role': 'user', 'content': 'Weather in Paris? 😀 héllo'},
                     {'role': 'assistant', 'tokens': TURN_2}, {'role': 'user', 'content': 'Thanks'},
                     {'role': 'assistant', 'tokens': TURN_1}, {'role': 'tool', 'name': 'get_weather', 'content': '{"temperature": 20}'}]
            self.assertEqual(h.history(rows[2]).ids, h.completion_prompt(h.render_turns(turns, CONFIG, tools=TOOLS)) + TURN_2)
            row = {'sample_id': 's', 'messages': history, 'generation_records': 3}
            attach_histories([row], Path(folder) / 'c' / 'requests.jsonl')
            self.assertEqual(row['history']['generation_index'], 2)
            self.assertEqual(sorted(set(h.encode_record(row)['labels'].tolist())), [-1, 0, 1, 2, 3, 4])

    def test_context_stop_repeats_for_a_talking_client(self):
        """After the context limit, a user simulator that keeps replying gets the same stop, not an error."""
        first_prompt = h.completion_prompt(h.render_turns(
            [{'role': 'system', 'content': 'Solve the task.'}, {'role': 'user', 'content': 'Weather in Paris? 😀 héllo'}],
            CONFIG, tools=TOOLS))
        with tempfile.TemporaryDirectory() as folder:
            server = HarmonyChatServer(folder, FakeSampler([TURN_2]), config=CONFIG, max_tokens=512, model_name='m',
                                       context_limit=len(first_prompt) + len(TURN_2) + 3)
            with server.serving() as url:
                route = url + '/v1/chat/completions'
                status, first = post(route, {'model': 'm', 'messages': MESSAGES, 'tools': TOOLS})
                self.assertEqual(status, 200, first)
                history = MESSAGES + [{'role': 'assistant', 'content': 'It is 20C.'}, {'role': 'user', 'content': 'Thanks'}]
                status, second = post(route, {'model': 'm', 'messages': history, 'tools': TOOLS})
                self.assertEqual((status, second['choices'][0]['message']['content']), (200, ''))
                history += [{'role': 'assistant', 'content': ''}, {'role': 'user', 'content': 'Hello?'}]
                status, third = post(route, {'model': 'm', 'messages': history, 'tools': TOOLS})
                self.assertEqual((status, third['choices'][0]['message']['content']), (200, ''))
            stops = (Path(folder) / 'episode_stops.jsonl').read_text().splitlines()
            self.assertEqual(len(stops), 2)
            row = {'sample_id': 's', 'messages': history, 'generation_records': 3}
            attach_histories([row], Path(folder) / 'requests.jsonl')
            self.assertEqual(row['history']['generation_index'], 0)


if __name__ == '__main__':
    unittest.main()
