"""Upstream standalone completion must not masquerade as a successful demo pick."""
from types import SimpleNamespace

import pytest

from ffw_pick.pick_moveit import pick_result_code


@pytest.mark.parametrize('status', ['stop', 'no path', 'empty hand', 'stuck', 'basket empty'])
def test_demo_failure_never_succeeds_in_preview_or_execution(status):
    hooks = SimpleNamespace(error_code='', message='')
    assert pick_result_code(status, hooks) == 1
    if status == 'basket empty':
        assert hooks.error_code == 'OBJECT_NOT_FOUND'
    if status == 'stuck':
        assert hooks.error_code == 'PICK_PLAN_FAILED'


def test_success_clears_previous_object_search_failure():
    hooks = SimpleNamespace(error_code='PICK_PLAN_FAILED', message='First object unreachable')
    assert pick_result_code('ok', hooks) == 0
    assert hooks.error_code == hooks.message == ''


def test_standalone_empty_basket_is_normal_completion():
    assert pick_result_code('basket empty') == 0
    assert pick_result_code('stop') == 1


def test_stop_preserves_specific_fault_and_holding_state():
    hooks = SimpleNamespace(error_code='SCENE_UNCONFIRMED', message='Scene update unknown',
                            holding=True, stop_confirmed=False)
    assert pick_result_code('stop', hooks) == 1
    assert hooks.error_code == 'SCENE_UNCONFIRMED'
    assert hooks.holding and not hooks.stop_confirmed
