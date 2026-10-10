import pytest

from auth import UserStore

PASSWORD = "S3gretissima!"


@pytest.fixture
def store():
    s = UserStore()
    s.register("anna", PASSWORD)
    return s


def test_login_ok(store):
    assert store.login("anna", PASSWORD) is True


def test_login_wrong_password(store):
    assert store.login("anna", "sbagliata123") is False


def test_login_unknown_user(store):
    assert store.login("nessuno", PASSWORD) is False


def test_duplicate_username(store):
    with pytest.raises(ValueError):
        store.register("anna", "altraPassword1")


def test_empty_username():
    with pytest.raises(ValueError):
        UserStore().register("", PASSWORD)


def test_short_password():
    with pytest.raises(ValueError):
        UserStore().register("bruno", "corta")


def test_password_is_not_stored_in_plain_text(store):
    assert PASSWORD not in repr(vars(store))


def test_change_password(store):
    store.change_password("anna", PASSWORD, "NuovaPassword9")
    assert store.login("anna", "NuovaPassword9") is True
    assert store.login("anna", PASSWORD) is False


def test_change_password_wrong_old(store):
    with pytest.raises(ValueError):
        store.change_password("anna", "nonQuella99", "NuovaPassword9")
    assert store.login("anna", PASSWORD) is True


def test_change_password_too_short(store):
    with pytest.raises(ValueError):
        store.change_password("anna", PASSWORD, "corta")
