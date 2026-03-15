from django import template

register = template.Library()


@register.filter
def pct(value: float | int) -> float:
    return float(value) * 100


@register.filter
def humanize_token(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split()).title()


@register.filter
def join_humanized(values) -> str:
    if not values:
        return ""
    return ", ".join(humanize_token(value) for value in values)
