import sys
from tensorboard.backend.event_processing import event_accumulator

if len(sys.argv) < 2:
    print('Usage: inspect_tb.py <event_file>')
    sys.exit(2)

path = sys.argv[1]
try:
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()
    print('TAGS:')
    for k, v in tags.items():
        try:
            vals = list(v)[:20]
        except Exception:
            vals = v
        print(k + ':', vals)
    print('\nSample scalars (first 5 each):')
    scalars = tags.get('scalars', [])
    for s in scalars:
        vals = ea.Scalars(s)
        print(f"{s}: count={len(vals)}")
        for item in vals[:5]:
            print('  ', item.step, item.value)
except Exception as e:
    print('ERROR reading event file:', e)
    sys.exit(1)
