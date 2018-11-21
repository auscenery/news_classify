# -*- coding: utf-8 -*-

import logging
from functools import wraps

from flask import request, make_response


_LOGGER = logging.getLogger(__name__)


def add_response_headers(headers={}):
    """This decorator adds the headers passed in to the response"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            resp = make_response(f(*args, **kwargs))
            req_h = request.headers
            h = resp.headers
            for header, value in headers.items():
                h[header] = value
            h['Access-Control-Allow-Origin'] = req_h.get('Origin', '*')
            h['Access-Control-Allow-Credentials'] = 'true'
            accept_header = req_h.get('Access-Control-Request-Headers')
            if accept_header:
                h['Access-Control-Allow-Headers'] = accept_header
            return resp
        return decorated_function
    return decorator


def access_control(f):
    @wraps(f)
    # 允许跨域
    @add_response_headers({
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'text/json;charset=utf-8',
    })
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated_function
