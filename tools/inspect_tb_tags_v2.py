import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = os.path.join('models','20260223155440','tb_logs')
for root, dirs, files in os.walk(base):
    for f in files:
        if not f.startswith('events.out.tfevents'):
            continue
        path = os.path.join(root, f)
        try:
            ea = EventAccumulator(path)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            print('FILE:', path)
            for t in tags:
                print('  ', t)
        except Exception as e:
            print('ERR reading', path, e)
