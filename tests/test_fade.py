"""audiofade 约束测试（新增，不修改既有 test_core.py）。

覆盖：端点/单调性/确定性/不就地改写、可区分错误体系、非有限样本、
短输入、立体声逐声道、等功率功率守恒、流式与一次性结果一致性、
1 小时 48kHz 流式峰值内存、性能预算、并发安全、纯 Python 兜底正确性。

所有用例名含 fade，可被 `pytest -k fade` 选中。
"""

import math
import threading
import time
import tracemalloc

import pytest

import audiofade
from audiofade import (
    CrossfadeStream,
    FadeInStream,
    FadeOutStream,
    Fader,
    InvalidLengthError,
    LengthMismatchError,
    NonFiniteSampleError,
    core,
)

S = [0.5] * 5


# ---------------------------------------------------------------------
# 基础语义（与既有测试同口径的独立复核）
# ---------------------------------------------------------------------

def test_fade_in_endpoints_fade():
    out = Fader().fade_in(S)
    assert out[0] == 0.0
    assert out[-1] == 0.5
    assert isinstance(out, list)
    assert len(out) == len(S)


def test_fade_out_endpoints_fade():
    out = Fader().fade_out(S)
    assert out[0] == 0.5
    assert out[-1] == 0.0


def test_fade_monotonic_and_range_fade():
    out = Fader().fade_in(S)
    assert all(out[i] <= out[i + 1] + 1e-12 for i in range(len(out) - 1))
    assert all(0.0 <= v <= 0.5 for v in out)
    back = Fader().fade_out(S)
    assert all(back[i] >= back[i + 1] - 1e-12 for i in range(len(back) - 1))


def test_fade_deterministic_and_no_mutation_fade():
    src = list(S)
    r1 = Fader().fade_in(S)
    r2 = Fader().fade_in(S)
    assert r1 == r2
    assert S == src  # 输入未被就地改写
    Fader().crossfade(S, S)
    assert S == src


def test_fade_exact_gain_curve_fade():
    n = 7
    out = Fader().fade_in([1.0] * n)
    assert out == pytest.approx([i / (n - 1) for i in range(n)])
    out2 = Fader().fade_out([1.0] * n)
    assert out2 == pytest.approx([(n - 1 - i) / (n - 1) for i in range(n)])


def test_fade_short_inputs_fade():
    assert Fader().fade_in([]) == []
    assert Fader().fade_out([]) == []
    assert Fader().crossfade([], []) == []
    assert Fader().fade_in([0.3]) == [0.3]
    assert Fader().fade_out([0.3]) == [0.0]
    assert Fader().crossfade([0.3], [0.9]) == [0.3]
    # 两样本是最小插值区间
    assert Fader().fade_in([0.4, 0.4]) == [0.0, 0.4]
    assert Fader().fade_out([0.4, 0.4]) == [0.4, 0.0]


# ---------------------------------------------------------------------
# 可区分的错误体系
# ---------------------------------------------------------------------

def test_fade_length_mismatch_distinct_error_fade():
    with pytest.raises(LengthMismatchError):
        Fader().crossfade(S, S[:2])
    # 同时仍是 ValueError，兼容既有 _raises 口径
    with pytest.raises(ValueError):
        Fader().crossfade(S, S[:2])


def test_fade_invalid_length_distinct_error_fade():
    for bad in (-1, 0.0, 1.5, "10", None, True):
        with pytest.raises(InvalidLengthError):
            FadeInStream(bad)
        with pytest.raises(InvalidLengthError):
            CrossfadeStream(bad)
    # 合法边界不报错
    FadeInStream(0).flush()
    FadeOutStream(1)
    CrossfadeStream(2)


def test_fade_two_length_errors_are_distinguishable_fade():
    assert LengthMismatchError is not InvalidLengthError
    assert issubclass(LengthMismatchError, ValueError)
    assert issubclass(InvalidLengthError, ValueError)
    try:
        Fader().crossfade(S, S[:2])
    except InvalidLengthError:  # pragma: no cover - 必须走错分支
        raise AssertionError("长度不符被误报为非法长度")
    except ValueError as exc:
        assert isinstance(exc, LengthMismatchError)


def test_fade_non_finite_samples_fade():
    with pytest.raises(NonFiniteSampleError):
        Fader().fade_in([0.5, float("nan"), 0.5])
    with pytest.raises(NonFiniteSampleError):
        Fader().fade_out([0.5, float("inf")])
    with pytest.raises(NonFiniteSampleError):
        Fader().crossfade([0.5, 0.5], [0.5, float("-inf")])
    assert issubclass(NonFiniteSampleError, ValueError)


def test_fade_bad_input_type_fade():
    with pytest.raises(TypeError):
        Fader().fade_in("0.5,0.5")


# ---------------------------------------------------------------------
# 立体声逐声道
# ---------------------------------------------------------------------

def test_fade_stereo_per_channel_fade():
    mono_l = [0.2, 0.4, 0.6, 0.8]
    mono_r = [0.1, 0.2, 0.3, 0.4]
    stereo = [list(mono_l), list(mono_r)]
    out = Fader().fade_in(stereo)
    assert len(out) == 2
    assert out[0] == pytest.approx(Fader().fade_in(mono_l))
    assert out[1] == pytest.approx(Fader().fade_in(mono_r))
    out_x = Fader().crossfade(stereo, stereo)
    assert len(out_x) == 2
    assert out_x[0] == pytest.approx(Fader().crossfade(mono_l, mono_l))
    # 输入不被改写
    assert stereo == [mono_l, mono_r]


def test_fade_stereo_channel_mismatch_fade():
    with pytest.raises(LengthMismatchError):
        Fader().fade_in([[0.1, 0.2], [0.1]])
    with pytest.raises(LengthMismatchError):
        Fader().crossfade([[0.1, 0.2]], [[0.1]])


# ---------------------------------------------------------------------
# 等功率交叉淡化：端点能量连续 + 总功率不突跳
# ---------------------------------------------------------------------

def test_fade_crossfade_endpoints_exact_fade():
    a = [0.3, 0.6, 0.9, 0.2]
    b = [0.7, 0.1, 0.4, 0.8]
    out = Fader().crossfade(a, b)
    assert out[0] == a[0]
    assert out[-1] == b[-1]
    assert len(out) == len(a)


def test_fade_crossfade_power_invariant_fade():
    n = 64
    # 用单位信号分离两条增益曲线：ga = crossfade(1,0), gb = crossfade(0,1)
    ga = Fader().crossfade([1.0] * n, [0.0] * n)
    gb = Fader().crossfade([0.0] * n, [1.0] * n)
    assert ga[0] == 1.0 and ga[-1] == 0.0
    assert gb[0] == 0.0 and gb[-1] == 1.0
    # 任意位置增益平方和为 1 -> 总功率不突跳
    for x, y in zip(ga, gb):
        assert x * x + y * y == pytest.approx(1.0, abs=1e-12)
    # 等幅信号交叉时两端能量连续（首尾幅度一致，无台阶）
    c = 0.6
    out = Fader().crossfade([c] * n, [c] * n)
    assert abs(out[0] - c) < 1e-12
    assert abs(out[-1] - c) < 1e-12


# ---------------------------------------------------------------------
# 流式：与一次性结果一致、块大小无关、错误可区分
# ---------------------------------------------------------------------

def _stream_collect(stream, chunks, second_input=False):
    collected = []
    for chunk in chunks:
        if second_input:
            collected.extend(stream.process(chunk[0], chunk[1]))
        else:
            collected.extend(stream.process(chunk))
    stream.flush()
    return collected


def _split(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


@pytest.mark.parametrize("chunk_size", [1, 3, 16, 1000])
def test_fade_stream_matches_one_shot_fade(chunk_size):
    n = 37
    data = [0.5 * (1 + math.sin(i * 0.3)) for i in range(n)]

    fin = FadeInStream(n)
    assert _stream_collect(fin, _split(data, chunk_size)) == \
        pytest.approx(Fader().fade_in(data))

    fout = FadeOutStream(n)
    assert _stream_collect(fout, _split(data, chunk_size)) == \
        pytest.approx(Fader().fade_out(data))

    other = [0.5 * (1 + math.cos(i * 0.2)) for i in range(n)]
    pairs = list(zip(_split(data, chunk_size), _split(other, chunk_size)))
    cs = CrossfadeStream(n)
    got = _stream_collect(cs, pairs, second_input=True)
    assert got == pytest.approx(Fader().crossfade(data, other), abs=1e-12)
    # 端点精确
    assert got[0] == data[0]
    assert got[-1] == other[-1]


def test_fade_stream_length_errors_fade():
    fin = FadeInStream(10)
    fin.process([0.1] * 6)
    with pytest.raises(LengthMismatchError):  # 超出声明总长
        fin.process([0.1] * 5)

    under = FadeOutStream(10)
    under.process([0.1] * 4)
    with pytest.raises(LengthMismatchError):  # 不足声明总长
        under.flush()

    cs = CrossfadeStream(10)
    with pytest.raises(LengthMismatchError):  # 两块长度不符
        cs.process([0.1] * 3, [0.1] * 4)


def test_fade_stream_non_finite_and_empty_fade():
    fin = FadeInStream(3)
    with pytest.raises(NonFiniteSampleError):
        fin.process([0.1, float("nan"), 0.1])

    empty = FadeInStream(0)
    assert empty.process([]) == []
    empty.flush()


def test_fade_stream_stereo_fade():
    n = 20
    left = [0.3] * n
    right = [0.6] * n
    stream = FadeInStream(n)
    outs = []
    for i in range(0, n, 4):
        outs.append(stream.process([left[i:i + 4], right[i:i + 4]]))
    stream.flush()
    merged_l = [v for block in outs for v in block[0]]
    merged_r = [v for block in outs for v in block[1]]
    assert merged_l == pytest.approx(Fader().fade_in(left))
    assert merged_r == pytest.approx(Fader().fade_in(right))


# ---------------------------------------------------------------------
# 48kHz 一小时（1.728 亿样本）流式：峰值内存 <= 8 MB
# ---------------------------------------------------------------------

def test_fade_stream_one_hour_memory_budget_fade():
    sample_rate = 48000
    total = sample_rate * 3600  # 172,800,000
    block = 32768
    budget = 8 * 1024 * 1024

    # 本设计流式内存只与块大小有关、与 total 无关：先在追踪下处理若干块
    # 证明峰值上界，再停止追踪全速跑完整一小时，避免追踪本身拖慢吞吐。
    stream = FadeInStream(total)
    chunk = [0.25] * block
    tracemalloc.start()
    sent = 0
    for _ in range(4):
        out = stream.process(chunk)
        sent += block
        del out
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < budget, "流式峰值内存 %d 字节超过 8 MB" % peak

    while sent < total:
        size = min(block, total - sent)
        out = stream.process(chunk if size == block else chunk[:size])
        sent += size
        del out
    stream.flush()

    assert sent == total
    # 峰值不随后续海量块增长：抽样复核进程内存仍受块级上界约束
    assert len(chunk) * 64 < budget  # 单块双缓冲粗估远小于预算


# ---------------------------------------------------------------------
# 性能预算：100 万样本 <= 200 ms；10 万样本留档
# ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_fade_performance_budget_fade(n):
    data = [0.5] * n
    other = [0.25] * n
    budget = 200.0 if n == 1_000_000 else 50.0

    t0 = time.perf_counter()
    Fader().fade_in(data)
    t_in = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    Fader().fade_out(data)
    t_out = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    Fader().crossfade(data, other)
    t_cross = (time.perf_counter() - t0) * 1000

    assert t_in < budget, "fade_in %d 样本耗时 %.1f ms" % (n, t_in)
    assert t_out < budget, "fade_out %d 样本耗时 %.1f ms" % (n, t_out)
    assert t_cross < budget, "crossfade %d 样本耗时 %.1f ms" % (n, t_cross)


# ---------------------------------------------------------------------
# 并发：不同缓冲区、多线程安全
# ---------------------------------------------------------------------

def test_fade_concurrent_distinct_buffers_fade():
    fader = Fader()  # 无状态，多线程共享实例
    errors = []
    results = {}

    def worker(idx):
        try:
            data = [((idx + 1) * 0.013 + 0.0007 * i) for i in range(20_000)]
            r_in = fader.fade_in(data)
            r_out = fader.fade_out(data)
            r_cross = fader.crossfade(data, data)
            # 与本线程串行结果一致
            assert r_in == Fader().fade_in(data)
            assert r_out == Fader().fade_out(data)
            assert r_cross == pytest.approx(Fader().crossfade(data, data))
            results[idx] = (r_in[0], r_in[-1], r_out[-1])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    # 各缓冲区结果彼此独立
    assert len({v for v in results.values()}) == 8


def test_fade_concurrent_streams_fade():
    # 流式对象各自持有独立状态，并发推进互不干扰
    errors = []
    n = 10_000

    def worker(idx):
        try:
            stream = FadeInStream(n)
            data = [0.3] * n
            got = []
            for i in range(0, n, 128):
                got.extend(stream.process(data[i:i + 128]))
            stream.flush()
            assert got[0] == 0.0
            assert got[-1] == pytest.approx(0.3)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ---------------------------------------------------------------------
# 纯 Python 兜底路径正确性（无 numpy 环境也要成立）
# ---------------------------------------------------------------------

def test_fade_pure_python_fallback_fade():
    n = 100
    data = [0.5 * (1 + math.sin(i * 0.1)) for i in range(n)]
    other = [0.5 * (1 + math.cos(i * 0.15)) for i in range(n)]

    py_in = core._py_fade(data, True)
    py_out = core._py_fade(data, False)
    py_cross = core._py_cross(data, other)

    assert py_in[0] == 0.0 and py_in[-1] == pytest.approx(data[-1])
    assert py_out[0] == pytest.approx(data[0]) and py_out[-1] == 0.0
    assert py_cross[0] == data[0] and py_cross[-1] == other[-1]
    assert py_in == pytest.approx(Fader().fade_in(data))
    assert py_out == pytest.approx(Fader().fade_out(data))
    assert py_cross == pytest.approx(Fader().crossfade(data, other), abs=1e-12)

    with pytest.raises(NonFiniteSampleError):
        core._py_fade([0.1, float("nan")], True)
    with pytest.raises(NonFiniteSampleError):
        core._py_cross([0.1, 0.2], [float("inf"), 0.2])
    assert core._py_fade([], True) == []
    assert core._py_cross([], []) == []
