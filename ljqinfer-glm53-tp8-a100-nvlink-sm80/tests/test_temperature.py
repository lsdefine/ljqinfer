import pytest
import torch
from model.sampling import pick_tokens
from test_service_template_contract import service
from server.service import ServiceError
from server.openai_protocol import to_service_request

@pytest.mark.parametrize('value', [None, 0, 1])
@pytest.mark.parametrize('stream', [False, True])
def test_temperature_forward(service, value, stream):
    body = {'messages':[{'role':'user','content':'hello'}], 'max_tokens':32}
    if value is not None: body['temperature'] = value
    request = to_service_request(body)
    assert service.intake(request)['temperature'] == (1 if value is None else value)
    if stream: list(service.stream(request))
    else: service.generate(request)
    assert service._strategy.temperature == (1 if value is None else value)

@pytest.mark.parametrize('value', [True, False, None, '1', -1, 0.5, 2, float('nan')])
def test_invalid_temperature(service, value):
    with pytest.raises(ServiceError, match='temperature'):
        service.intake({'messages':[{'role':'user','content':'hello'}], 'temperature':value})

def test_mixed_sampling_distribution():
    torch.manual_seed(43)
    logits = torch.tensor([0.2, 0.3, 0.5]).log().expand(2, 20000, 3)
    tokens = pick_tokens(logits, [0, 1])
    assert (tokens[0] == 2).all()
    frequencies = tokens[1].bincount(minlength=3).float()/20000
    torch.testing.assert_close(frequencies, torch.tensor([0.2, 0.3, 0.5]), atol=.015, rtol=0)


def test_nucleus_tail_and_greedy():
    scores = torch.tensor([.80, .16, .03, .01]).log().expand(2, 10000, 4)
    result = pick_tokens(scores, [0, 1])
    assert (result[0] == 0).all()
    assert (result[1] < 2).all()
    freq = result[1].bincount(minlength=4).float() / 10000
    torch.testing.assert_close(freq[:2], torch.tensor([.80, .16]) / .96, atol=.02, rtol=0)
    assert (pick_tokens(scores, [1, 1], top_p=.01) == 0).all()
    assert (pick_tokens(scores, [1, 1], top_p=1) >= 2).any()


@pytest.mark.parametrize('p', [0, -1, 1.1, float('nan'), True])
def test_invalid_top_p(p):
    with pytest.raises(ValueError, match='top_p'):
        pick_tokens(torch.zeros(1, 4), [1], top_p=p)
