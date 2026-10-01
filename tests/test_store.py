import os
import tempfile

import pytest


@pytest.fixture
def store(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    os.remove(path)  # let store create it fresh
    from config import Config
    monkeypatch.setattr(Config, "APP_DB_PATH", path)
    import importlib
    import store as store_module
    importlib.reload(store_module)
    store_module.init_db()
    yield store_module
    os.remove(path)


def test_append_and_get_turns(store):
    store.append_turn("tenantA", "s1", "user", "hello")
    store.append_turn("tenantA", "s1", "assistant", "hi there")
    turns = store.get_turns("tenantA", "s1")
    assert turns == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]


def test_tenant_isolation(store):
    store.append_turn("tenantA", "s1", "user", "a-message")
    store.append_turn("tenantB", "s1", "user", "b-message")
    assert store.get_turns("tenantA", "s1") == [{"role": "user", "content": "a-message"}]
    assert store.get_turns("tenantB", "s1") == [{"role": "user", "content": "b-message"}]


def test_clear_session(store):
    store.append_turn("tenantA", "s1", "user", "hello")
    store.clear_session("tenantA", "s1")
    assert store.get_turns("tenantA", "s1") == []


def test_indexing_status_roundtrip(store):
    store.set_indexing_status("tenantA", "/a.pdf", "processing", 50)
    status = store.get_indexing_status("tenantA", "/a.pdf")
    assert status["status"] == "processing"
    assert status["progress"] == 50

    store.set_indexing_status("tenantA", "/a.pdf", "completed", 100)
    status = store.get_indexing_status("tenantA", "/a.pdf")
    assert status["status"] == "completed"


def test_indexing_status_unknown_file(store):
    assert store.get_indexing_status("tenantA", "/never-seen.pdf") == {"status": "unknown"}
