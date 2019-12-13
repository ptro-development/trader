from __future__ import absolute_import
from celery import Celery
from celery.bin import Option
from trader import celeryconfig

app = Celery('trader')
app.config_from_object(celeryconfig)

app.user_options['trader'].add(
    Option('--enable_replay_mode', action='store_true', default=False,
            help='To enable mode for replaying of historic trade data processing'),  # noqa
    )


# from celery import bootsteps
"""
class MyBootstep(bootsteps.Step):

    def __init__(self, worker, enable_replay_mode=False, **options):
        print "AAAAAAA ", enable_replay_mode
        with open("/tmp/qwe", "w") as f:
            f.write(str("AAAAAAA " + str( enable_replay_mode) + "\n"))

    def start(self, parent):
        # our step is started together with all other Worker/Consumer
        # bootsteps.
        print('{0!r} is starting'.format(parent))
"""

if __name__ == '__main__':
    # app.steps['trader'].add(MyBootstep)
    app.start()
