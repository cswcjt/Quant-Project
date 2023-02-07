from django import template

register = template.Library()


@register.filter
def convert_blank_to_underline(value):
    return '_'.join(value.split(' '))