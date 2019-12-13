from trader.celery import app


def should_terminate(function_name):
    return app.conf.SHOULD_TERMINATE.get(function_name, False)
