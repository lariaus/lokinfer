"""report.html generation and the mlogs-viewer HTTP API on a synthetic run directory."""
import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lokinfer.prof import analysis, report, viewer

from test_prof_analysis import FINAL_IR, RAW  # noqa: E402 (same directory)


def make_run(base, rid, e2e=1.0):
    d = base / rid
    (d / "ir").mkdir(parents=True)
    (d / "ir" / "00-source.mlir").write_text("source\n")
    (d / "ir" / "01-final.mlir").write_text(FINAL_IR)
    passes = [{"name": "source", "text": "source\n", "file": "ir/00-source.mlir"},
              {"name": "final", "text": FINAL_IR, "file": "ir/01-final.mlir"}]
    run = analysis.build_run(copy.deepcopy(RAW), meta={"id": rid, "model": "M", "case": "c", "device": "Mac GPU",
                                                       "target": "mac", "backend": "gpu", "created": rid},
                             e2e_ms=[e2e], ir_passes=passes)
    run["notes"] = ["note </script> with a closing tag"]
    (d / "run.json").write_text(json.dumps(run))
    return d


def test_report_embeds_run_and_ir(tmp_path):
    d = make_run(tmp_path, "r1")
    out = report.write_report(d)
    html = out.read_text()
    assert "</script> with" not in html  # the payload can't close the script element
    start = html.index("/*EMBEDDED_DATA*/") + len("/*EMBEDDED_DATA*/")
    payload = json.loads(html[start:html.index("</script>", start)])
    assert payload["run"]["id"] == "r1" and payload["files"]["ir/01-final.mlir"] == FINAL_IR
    assert "<title>M Mac GPU profile</title>" in html
    assert report.APP.read_text().count("/*EMBEDDED_DATA*/null") == 1  # the template itself is untouched


@pytest.fixture()
def server(tmp_path):
    make_run(tmp_path, "20261101-000001-a", 1.0)
    make_run(tmp_path, "20261101-000002-b", 2.0)
    (tmp_path / "not-a-run").mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), viewer.make_handler(tmp_path))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_viewer_api(server):
    st, ct, body = get(server + "/")
    assert st == 200 and ct.startswith("text/html") and b"/*EMBEDDED_DATA*/null" in body
    runs = json.loads(get(server + "/api/runs")[2])
    assert [r["id"] for r in runs] == ["20261101-000002-b", "20261101-000001-a"]
    assert runs[0]["e2e_ms"] == 2.0 and runs[0]["kernel_calls"] == 3
    run = json.loads(get(server + "/api/run/20261101-000001-a")[2])
    assert run["kind"] == "lokinfer"
    assert get(server + "/api/run/20261101-000001-a/file/ir/01-final.mlir")[2].decode() == FINAL_IR
    c = json.loads(get(server + "/api/compare?a=20261101-000001-a&b=20261101-000002-b")[2])
    assert c["e2e"]["ratio"] == pytest.approx(2.0) and c["kernels_matched"] == 3


@pytest.mark.parametrize("path", ["/api/run/../../etc/passwd", "/api/run/20261101-000001-a/file/../../x",
                                  "/runs/../secret", "/api/run/nope", "/api/compare?a=x&b=y", "/nothing"])
def test_viewer_rejects_bad_paths(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + path)
    assert e.value.code == 404


def test_url_for():
    assert viewer.url_for(8766) == "http://127.0.0.1:8766/"
    assert viewer.url_for(8766, "a b").endswith("/runs/a%20b/report.html")
    assert viewer.url_for(8766, compare=("x", "y")).endswith("#a=x&b=y")
