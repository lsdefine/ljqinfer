"""Full text, width 1 versus width 4, character for character."""
import threading
from tests.mtp_consistency import call, Q

def run(b):
    out = {}
    th = [threading.Thread(target=call, args=(out, i, Q, 96)) for i in range(b)]
    [t.start() for t in th]
    [t.join() for t in th]
    return out

a = run(1)[0]
c = run(4)
print('WIDTH1 mtp=%.3f len=%d' % (a[1].get('mtp_accepted_per_step', -1), len(a[0])), flush=True)
for i in sorted(c):
    print('W4[%d] mtp=%.3f len=%d same_as_alone=%s'
          % (i, c[i][1].get('mtp_accepted_per_step', -1), len(c[i][0]), c[i][0] == a[0]), flush=True)
print('ALONE: %r' % a[0], flush=True)
print('BATCH: %r' % c[0][0], flush=True)
