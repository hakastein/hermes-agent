"""Long-lived agent/plugin subprocess clients die with the Hermes process that started them.

Each kill -9 case drives the REAL production spawn path in two disposable owner processes, SIGKILLs
one (no atexit, no finally: nothing of ours runs) and requires that owner's child tree to be gone
while the live owner's tree is untouched. On origin/main every one of these trees survived kill -9
forever: plain Popen children reparent to init, and detached ones lead their own session.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")

REPO = Path(__file__).resolve().parents[2]

# The child each production path starts: records itself, starts a grandchild that stays in its
# process group (the way a node bridge's helpers or a dev server's workers do), then idles.
_CHILD = r'''
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
out = Path(sys.argv[1]); role = sys.argv[2]
p = psutil.Process()
(out / (role + '.tmp')).write_text(json.dumps({'pid': p.pid, 'created': p.create_time()}))
(out / (role + '.tmp')).replace(out / (role + '.json'))
if role == 'child':
    subprocess.Popen([sys.executable, __file__, str(out), 'grandchild'], stdin=subprocess.DEVNULL)
time.sleep(120)
'''

# Owner bodies: ``child`` is the argv starting the child, ``child_cmd`` the same as a shell string,
# ``out`` the receipt directory. Each starts the child through the production path; the owner idles.
_OWNERS = {
    "whatsapp-bridge": (
        "from plugins.platforms.whatsapp.adapter import _spawn_bridge_process\n"
        "_spawn_bridge_process(child, open(out / 'bridge.log', 'a', encoding='utf-8'), dict(os.environ))\n"
    ),
}


def _owner_source(site: str) -> str:
    return (
        "import os, shlex, sys, threading, time\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[1]); child_script = sys.argv[2]\n"
        "child = [sys.executable, child_script, str(out), 'child']\n"
        "child_cmd = ' '.join(shlex.quote(a) for a in child)\n"
        + _OWNERS[site]
        + "time.sleep(300)\n"
    )


def _read(path: Path) -> dict:
    wait_until(path.exists, timeout=30, what=f"receipt {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _alive(identity: dict) -> bool:
    """Zombie-aware: a killed orphan lingers as a zombie until init reaps it."""
    try:
        proc = psutil.Process(identity["pid"])
        return proc.create_time() == identity["created"] and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _start_owner(tmp_path: Path, name: str, site: str) -> tuple[subprocess.Popen, Path]:
    out = tmp_path / name
    out.mkdir()
    child_script = tmp_path / "child.py"
    child_script.write_text(_CHILD, encoding="utf-8")
    home = tmp_path / f"home-{name}"
    home.mkdir()
    env = dict(os.environ, HERMES_HOME=str(home / ".hermes"), HOME=str(home), HERMES_REAL_HOME=str(home),
               PYTHONPATH=str(REPO))
    with (tmp_path / f"{name}.err").open("w", encoding="utf-8") as err:
        owner = subprocess.Popen([sys.executable, "-c", _owner_source(site), str(out), str(child_script)],
                                 env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
    return owner, out


# The children reparent to init once their owner dies, so cleaning them up leaves this test's
# subtree; real signal delivery to a real orphan is the point of the test.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("site", sorted(_OWNERS))
def test_a_sigkilled_owner_takes_its_child_tree_and_spares_a_live_owners(tmp_path, site):
    owners: dict = {}
    ids: dict = {}
    try:
        for name in ("dead", "live"):
            owners[name], out = _start_owner(tmp_path, name, site)
            ids[name] = {role: _read(out / f"{role}.json") for role in ("child", "grandchild")}
        assert all(_alive(i) for tree in ids.values() for i in tree.values())

        owners["dead"].kill()  # SIGKILL: no atexit, no finally, nothing of ours runs
        owners["dead"].wait(timeout=10)

        wait_until(lambda: not any(_alive(i) for i in ids["dead"].values()), timeout=15,
                   what=f"the {site} tree to die with its kill -9'd owner")
        assert all(_alive(i) for i in ids["live"].values()), f"a live owner's {site} tree was killed"
    finally:
        for tree in ids.values():
            for identity in tree.values():
                if _alive(identity):
                    psutil.Process(identity["pid"]).kill()
        for owner in owners.values():
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=10)


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL,
                            start_new_session=True)


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


@pytest.mark.asyncio
async def test_whatsapp_disconnect_releases_the_bridge_it_stopped(monkeypatch):
    from plugins.platforms.whatsapp import adapter as wa

    released = []
    monkeypatch.setattr(wa, "release_process_groups", lambda pgids: released.append(list(pgids)))
    bridge = _sleeper()
    from tests.gateway.test_whatsapp_connect import _make_adapter

    adapter = _make_adapter()
    adapter._bridge_process, adapter._poll_task = bridge, None
    adapter._running, adapter._session_lock_identity = True, None
    try:
        await adapter.disconnect()
        assert bridge.poll() is not None, "disconnect() left the bridge running"
        assert released == [[bridge.pid]]
    finally:
        _reap(bridge)
