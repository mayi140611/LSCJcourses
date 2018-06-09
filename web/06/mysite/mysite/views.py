# -*- encoding=utf-8 -*-

from django.http.response import HttpResponse


def index(request):
    return HttpResponse('Hello World!  你好')