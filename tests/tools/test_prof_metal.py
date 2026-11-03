"""Parsers of the Apple tool outputs (metal-objdump AIR dumps, xctrace exports, sample(1) reports), on fixtures."""
from lokinfer.prof import metal

AIR_DUMP = """
/x/lokinfer.metallib:	file format metallib
Disassembly of section MODULE_LIST:

0x00000000003546 -- add_f32:
source_filename = "add_f32"
define void @add_f32(float addrspace(1)* %0, float addrspace(1)* %1, float addrspace(1)* %2, i32 %3) {
  %5 = load <4 x float>, <4 x float> addrspace(1)* %a, align 16
  %6 = load <4 x float>, <4 x float> addrspace(1)* %b, align 16
  %7 = fadd fast <4 x float> %6, %5
  store <4 x float> %7, <4 x float> addrspace(1)* %c, align 16
  ret void
}
!1 = !DIFile(filename: "/r/kernels/gpu/elementwise.metal", directory: "/r")

0x00000000004000 -- gemm_f32:
define void @gemm_f32() {
  br label %10
10:
  %11 = load float, float addrspace(3)* %t, align 4
  %12 = call float @llvm.fma.f32(float %11, float %11, float %11)
  tail call void @air.wg.barrier(i32 2, i32 1)
  store float %12, float addrspace(3)* %t, align 4
  br i1 %c, label %10, label %20, !llvm.loop !5
20:
  ret void
}
"""


def test_split_air_and_stats():
    mods = metal.split_air_dump(AIR_DUMP)
    assert list(mods) == ["add_f32", "gemm_f32"]
    a = metal.air_stats(mods["add_f32"])
    assert a["load_device"] == 2 and a["store_device"] == 1 and a["fadd"] == 1 and a["instructions"] == 5
    assert a["source"] == "/r/kernels/gpu/elementwise.metal"
    g = metal.air_stats(mods["gemm_f32"])
    assert g["load_threadgroup"] == 1 and g["store_threadgroup"] == 1 and g["fma"] == 1 and g["barrier"] == 1
    assert g["loops"] == 1 and g["blocks"] == 2 and g["source"] is None


def test_find_source_real_kernels():
    f, line = metal.find_source("add_f32")
    assert f.name == "elementwise.metal" and line > 0
    f, _ = metal.find_source("lk_conv1d_lc_f32")  # declared through [[host_name("...")]]
    assert f.name == "kokoro_lc.metal"
    assert metal.find_source("definitely_not_a_kernel") is None


TIME_PROFILE = """<?xml version="1.0"?>
<trace-query-result><node><schema name="time-profile">
<col><mnemonic>time</mnemonic></col><col><mnemonic>thread</mnemonic></col><col><mnemonic>process</mnemonic></col>
<col><mnemonic>weight</mnemonic></col><col><mnemonic>stack</mnemonic></col></schema>
<row><sample-time id="1">1</sample-time><thread id="2" fmt="Main Thread"/><process id="4" fmt="Python (12)"/>
<weight id="9" fmt="1.00 ms">1000000</weight>
<tagged-backtrace id="10"><frame id="11" name="lk::kernels::cpu::gemm"/><frame id="13" name="run"/><frame id="15" name="main"/></tagged-backtrace></row>
<row><sample-time id="16">2</sample-time><thread ref="2"/><process ref="4"/><weight ref="9"/><tagged-backtrace ref="10"/></row>
<row><sample-time id="17">3</sample-time><thread ref="2"/><process ref="4"/><weight ref="9"/>
<tagged-backtrace id="18"><frame id="19" name="write"/><frame ref="13"/><frame ref="15"/></tagged-backtrace></row>
<row><sample-time id="20">4</sample-time><thread ref="2"/><process id="21" fmt="kernel_task (0)"/><weight ref="9"/>
<tagged-backtrace id="22"><frame id="23" name="idle"/></tagged-backtrace></row>
</node></trace-query-result>"""


def test_time_profile_summary_resolves_refs():
    s = metal.time_profile_summary(TIME_PROFILE, process="python")
    assert s["samples"] == 3 and s["weight_ns"] == 3_000_000
    self_ = {r["symbol"]: r["ms"] for r in s["self"]}
    assert self_ == {"lk::kernels::cpu::gemm": 2.0, "write": 1.0}
    total = {r["symbol"]: r["ms"] for r in s["total"]}
    assert total["main"] == 3.0 and total["run"] == 3.0
    assert metal.time_profile_summary(TIME_PROFILE)["samples"] == 4


GPU_INTERVALS = """<?xml version="1.0"?>
<trace-query-result><node><schema name="metal-gpu-intervals">
<col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col><col><mnemonic>channel-name</mnemonic></col>
<col><mnemonic>process</mnemonic></col></schema>
<row><start-time id="1">1000</start-time><duration id="2">500</duration><gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name><process id="4" fmt="Python (3)"/></row>
<row><start-time id="5">1200</start-time><duration id="6">800</duration><gpu-channel-name ref="3"/><process ref="4"/></row>
<row><start-time id="7">5000</start-time><duration ref="2"/><gpu-channel-name id="8" fmt="Render">Render</gpu-channel-name><process id="9" fmt="WindowServer (1)"/></row>
</node></trace-query-result>"""


def test_interval_summary():
    s = metal.interval_summary(GPU_INTERVALS, "channel-name")
    assert s["count"] == 3 and s["total_ms"] == 1800 / 1e6
    assert {r["label"]: r["count"] for r in s["by_label"]} == {"Compute": 2, "Render": 1}
    assert s["busy_ms"] == (1000 + 500) / 1e6 and s["span_ms"] == 4500 / 1e6
    p = metal.interval_summary(GPU_INTERVALS, "channel-name", process="python")
    assert p["count"] == 2 and p["busy_ms"] == 1000 / 1e6


def test_counter_summary_and_format():
    xml = """<trace-query-result><node><schema><col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
    <col><mnemonic>name</mnemonic></col><col><mnemonic>percent-value</mnemonic></col></schema>
    <row><start-time>0</start-time><duration>10</duration><string id="1" fmt="ALU Utilization">x</string><percent>20</percent></row>
    <row><start-time>10</start-time><duration>30</duration><string ref="1"/><percent>60</percent></row>
    </node></trace-query-result>"""
    c = metal.counter_summary(xml)
    assert c["counters"] == [{"counter": "ALU Utilization", "mean": 50.0, "max": 60.0}]
    text = metal.format_trace_summary({"trace": "t.trace", "schemas": ["a"], "tables": {
        "metal-gpu-counter-intervals": c, "metal-gpu-intervals": metal.interval_summary(GPU_INTERVALS, "channel-name"),
        "time-profile": metal.time_profile_summary(TIME_PROFILE)}})
    assert "ALU Utilization" in text and "Compute" in text and "lk::kernels::cpu::gemm" in text


SAMPLE_REPORT = """Analysis of sampling Python (pid 1) every 1 millisecond
Call graph:
    100 Thread_1
Total number in stack (recursive counted multiple times):
Sort by top of stack, same collapsed (when >= 5):
        __psynch_cvwait  (in libsystem_kernel.dylib)        11823
        lk::kernels::cpu::gemm_bias_relu_f32(float const*)  (in _lokinfer.so)        1606
        tiny  (in x)        66

Binary Images:
"""


def test_sample_top_of_stack():
    rows = metal.sample_top_of_stack(SAMPLE_REPORT)
    assert rows[0] == ("__psynch_cvwait  (in libsystem_kernel.dylib)", 11823) and len(rows) == 3
    assert metal.sample_top_of_stack("no section") == []


def test_toc_schemas():
    toc = '<trace-toc><run number="1"><data><table schema="time-profile"/><table schema="tick" frequency="1"/>' \
          '<table schema="time-profile"/></data></run></trace-toc>'
    assert metal.toc_schemas(toc) == ["time-profile", "tick", "time-profile"]


COUNTER_INFO = """<trace-query-result><node><schema><col><mnemonic>timestamp</mnemonic></col><col><mnemonic>counter-id</mnemonic></col>
<col><mnemonic>name</mnemonic></col></schema>
<row><event-time id="1">0</event-time><uint32 id="2" fmt="0">0</uint32><gpu-counter-name id="3" fmt="ALU Utilization">ALU Utilization</gpu-counter-name></row>
<row><event-time ref="1"/><uint32 id="4" fmt="1">1</uint32><gpu-counter-name id="5" fmt="Compute Occupancy">Compute Occupancy</gpu-counter-name></row>
</node></trace-query-result>"""

# one row per line, like xctrace writes it; values and ids are de-duplicated with id / ref
COUNTER_VALUES = """<?xml version="1.0"?>
<trace-query-result>
<node><schema name="gpu-counter-value"><col><mnemonic>timestamp</mnemonic></col></schema>
<row><event-time id="1" fmt="0">0</event-time><uint32 id="2" fmt="0">0</uint32><fixed-decimal id="3" fmt="60">60.0</fixed-decimal><uint64 id="4">7</uint64></row>
<row><event-time ref="1"/><uint32 id="5" fmt="1">1</uint32><fixed-decimal id="6" fmt="0">0.0</fixed-decimal><uint64 ref="4"/></row>
<row><event-time id="7">10</event-time><uint32 ref="2"/><fixed-decimal id="8" fmt="20">20.0</fixed-decimal><uint64 ref="4"/></row>
<row><event-time ref="7"/><uint32 ref="5"/><fixed-decimal ref="3"/><uint64 ref="4"/></row>
</node></trace-query-result>
"""


def test_counter_values_summary(tmp_path):
    f = tmp_path / "v.xml"
    f.write_text(COUNTER_VALUES)
    s = metal.counter_values_summary(f, COUNTER_INFO)
    c = {r["counter"]: r for r in s["counters"]}
    assert s["rows"] == 4
    assert c["ALU Utilization"]["mean"] == 40.0 and c["ALU Utilization"]["max"] == 60.0
    occ = c["Compute Occupancy"]
    assert occ["mean"] == 30.0 and occ["mean_nonzero"] == 60.0 and occ["nonzero_pct"] == 50.0


def test_metric_summary():
    xml = """<trace-query-result><node><schema><col><mnemonic>timestamp</mnemonic></col><col><mnemonic>duration</mnemonic></col>
    <col><mnemonic>process</mnemonic></col><col><mnemonic>metric-value-int</mnemonic></col>
    <col><mnemonic>metric-value-double</mnemonic></col><col><mnemonic>metric-name</mnemonic></col></schema>
    <row><t>0</t><d>1</d><process id="3" fmt="Python (1)"/><uint64 id="6">0</uint64><f>3.0</f><string id="8" fmt="useful">useful</string></row>
    <row><t>0</t><d>1</d><process ref="3"/><uint64 ref="6"/><f>1.0</f><string id="9" fmt="processing">processing</string></row>
    <row><t>0</t><d>1</d><process ref="3"/><uint64>500</uint64><f>0</f><string id="10" fmt="cycle">cycle</string></row>
    <row><t>0</t><d>1</d><process id="11" fmt="other (2)"/><uint64 ref="6"/><f>9.0</f><string ref="8"/></row>
    </node></trace-query-result>"""
    m = {r["metric"]: r for r in metal.metric_summary(xml, "python")["metrics"]}
    assert m["useful"]["share_pct"] == 75.0 and m["processing"]["mean"] == 1.0 and m["cycle"]["int_sum"] == 500
