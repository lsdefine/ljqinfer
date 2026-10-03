"""Does the answer depend only on how wide the batch is?"""
import threading
from tests.mtp_consistency import call, Q

for b in (1, 2, 3, 4):
    out = {}
    th = [threading.Thread(target=call, args=(out, i, Q)) for i in range(b)]
    [t.start() for t in th]
    [t.join() for t in th]
    texts = {out[i][0] for i in range(b)}
    print('b=%d  same-within=%s' % (b, len(texts) == 1))
    for i in range(b):
        t, s = out[i]
        print('   r%d hit=%s chunks=%s pf=%s mtp=%.3f steps=%s  %r'
              % (i, s.get('cache_hit_tokens'), s.get('chunks'),
                 s.get('prefill_tokens'), s.get('mtp_accepted_per_step'),
                 s.get('decode_steps'), t[:48]))
