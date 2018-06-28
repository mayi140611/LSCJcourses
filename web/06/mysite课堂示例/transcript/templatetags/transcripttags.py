# -*- encoding=utf-8 -*-

from django import template


register = template.Library()


@register.filter(name='format_phone')
def format_phone(value):
    if len(value) == 11:
        return '{}-{}-{}'.format(value[:3], value[4:8], value[-4:])
    return value
