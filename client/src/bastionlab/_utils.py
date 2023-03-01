from __future__ import annotations
from typing import Callable, List, Optional


def delegate(
    target_cls: Callable,
    target_attr: str,
    f_names: List[str],
    wrap: bool = False,
    wrap_fn: Optional[Callable] = None,
    make_docstring: Optional[Callable[[str], str]] = None,
) -> Callable[[Callable], Callable]:
    def inner(cls: Callable) -> Callable:
        delegates = {f_name: getattr(target_cls, f_name) for f_name in f_names}

        def delegated_fn(f_name: str) -> Callable:
            def f(_self, *args, **kwargs):
                res = (delegates[f_name])(getattr(_self, target_attr), *args, **kwargs)
                if wrap:
                    if wrap_fn is not None:
                        return wrap_fn(_self, res)
                    else:
                        wrapped_res = _self.clone()
                        setattr(wrapped_res, target_attr, res)
                        return wrapped_res
                else:
                    return res

            return f

        for f_name in f_names:
            delegated = delegated_fn(f_name)
            if make_docstring is not None:
                delegated.__doc__ = make_docstring(f_name)

            setattr(cls, f_name, delegated)

        return cls

    return inner


def delegate_properties(
    *names: str, target_attr: str
) -> Callable[[Callable], Callable]:
    def inner(cls: Callable) -> Callable:
        def prop(name):
            def f(_self):
                return getattr(getattr(_self, target_attr), name)

            return property(f)

        for name in names:
            setattr(cls, name, prop(name))

        return cls

    return inner
