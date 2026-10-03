"""Image protocol/preprocessing oracle: released source is loaded independently."""
import base64
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from server.image_input import (decode_record, inspect_image, expand_images, patchify, validate_payload,
                                CONFIG, IMAGE_ID)

MODEL = Path('/mnt/data/kw/models/DeepSeek-V4.1-Flash')


def uri(size=(84,126), mode='RGB', fmt='PNG'):
    im = Image.new(mode, size, 100)
    buf = io.BytesIO(); im.save(buf, format=fmt)
    return 'data:image/'+fmt.lower()+';base64,'+base64.b64encode(buf.getvalue()).decode()


def official():
    spec = importlib.util.spec_from_file_location('released_image_processor', MODEL/'inference/image_processor.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('size,mode', [((1,1),'L'),((84,126),'RGBA'),((731,193),'RGB'),((5000,1),'RGB'),((1600,2000),'RGB')])
def test_preprocess_exact_released(size, mode):
    ref = official(); url = uri(size,mode)
    ids, payload = expand_images([10,IMAGE_ID,20], [{'url':url}])
    image = payload['images'][0]
    expected,h,w,nh,nw = ref.load_image({'url':url}, CONFIG)
    actual = patchify(image)
    import torch
    assert torch.equal(torch.from_numpy(actual).to(torch.bfloat16), expected.flatten(1))
    assert image['grid']==[h,w] and image['llm_grid']==[nh,nw]
    assert ids == [10]+[IMAGE_ID]*len(ref.image_token_types(nh,nw))+[20]
    validate_payload(ids,payload)


@pytest.mark.parametrize('url', ['https://example.org/a.png','file:///etc/passwd','/tmp/image.png',
    'data:image/png;base64,???','data:image/svg+xml;base64,PHN2Zz4=', 'data:image/png,abcd'])
def test_reject_unsupported(url):
    with pytest.raises(ValueError): inspect_image(decode_record({"url":url}))


def test_multi_order_and_spans():
    ids,p = expand_images([1,IMAGE_ID,2,IMAGE_ID,3], [{'url':uri()}, {'url':uri((42,42))}])
    a,b=p['images']; assert a['start']==1 and b['start']==2+a['length']
    assert ids[b['start']-1]==2 and ids[-1]==3
    validate_payload(ids,p)
    p['images'][1]['start']=1
    with pytest.raises(ValueError):validate_payload(ids,p)


def test_count_and_truncated():
    with pytest.raises(ValueError):expand_images([IMAGE_ID],[])
    with pytest.raises(ValueError):expand_images([], [{'url':uri()}])
    with pytest.raises(ValueError):inspect_image(decode_record({"url":uri()[:-9]}))


def test_service_preserves_history_images():
    from server.service import ServiceLayer
    from server.openai_protocol import to_service_request
    layer=ServiceLayer(SimpleNamespace(query=lambda *a,**k:None),tokenizer_path=str(MODEL/'tokenizer.json'))
    req={'messages':[{'role':'user','content':[{'type':'text','text':'before'},
        {'type':'image_url','image_url':{'url':uri()}},{'type':'text','text':'after'}]},
        {'role':'assistant','content':'ok'},{'role':'user','content':'what color?'}], 'max_tokens':8}
    ids,plan=layer.build(to_service_request(req))
    assert len(plan['image_payload']['images'])==1
    assert ids.count(IMAGE_ID)==plan['image_payload']['images'][0]['length']
    assert 'before' in layer.tokenizer.decode(ids) and 'after' in layer.tokenizer.decode(ids)


def test_server_image_module_does_not_import_torch():
    import subprocess
    r=subprocess.run([sys.executable,'-c',"import server.image_input,sys;assert 'torch' not in sys.modules"],capture_output=True)
    assert r.returncode==0,r.stderr
