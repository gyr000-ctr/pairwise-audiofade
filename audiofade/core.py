"""audiofade 核心 API。

增益约定
--------
- 线性淡入 ``fade_in``：增益 ``i/(N-1)``，从 0 线性增长到 1（首样本为 0、
  末样本等于原值）。
- 线性淡出 ``fade_out``：增益 ``(N-1-i)/(N-1)``，从 1 线性衰减到 0
  （首样本等于原值、末样本为 0）。
- 等功率交叉淡化 ``crossfade``：``theta = pi/2 * i/(N-1)``，
  ``out = a*cos(theta) + b*sin(theta)``。首样本严格等于 ``a[0]``、
  末样本严格等于 ``b[-1]``（两端能量连续），且
  ``cos(theta)^2 + sin(theta)^2 == 1``，混合全程总功率不突跳。

立体声
------
输入为嵌套序列 ``[左声道, 右声道]`` 时按声道逐通道施加同一组增益，
输出保持相同结构；扁平序列按单声道处理。

流式处理
--------
:class:`FadeInStream` / :class:`FadeOutStream` / :class:`CrossfadeStream`
按块推进，内部只保留旋转增益状态（与块大小无关的常数级开销），
不缓存全部样本，可流式处理 48kHz、1 小时（1.728 亿样本）信号。

错误体系（均可 ``except ValueError`` 捕获，且类型可区分）
--------------------------------------------------------
- :class:`LengthMismatchError`：长度不符（交叉淡化两端等长校验、
  流式累计样本数与声明总长不符）。
- :class:`InvalidLengthError`：非法长度（总长不是非负整数等）。
- :class:`NonFiniteSampleError`：样本含 NaN/Inf。
"""

import math

try:  # numpy 为可选加速热路径；缺失时纯 Python 实现保证功能完整
    import numpy as _np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - 环境无 numpy 时走兜底
    _np = None
    _HAVE_NUMPY = False

__all__ = [
    "Fader",
    "FadeError",
    "LengthMismatchError",
    "InvalidLengthError",
    "NonFiniteSampleError",
    "FadeInStream",
    "FadeOutStream",
    "CrossfadeStream",
]

_HALF_PI = math.pi / 2.0


class FadeError(ValueError):
    """所有淡化参数/数据错误的基类，本身仍是 ``ValueError``。"""


class LengthMismatchError(FadeError):
    """长度不符：交叉淡化两端长度不一致，或流式累计长度与声明不符。"""


class InvalidLengthError(FadeError):
    """非法长度：长度参数不是非负整数。"""


class NonFiniteSampleError(FadeError):
    """样本中存在 NaN 或无穷大。"""


def _validate_total(total):
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise InvalidLengthError(
            "total 必须是非负整数，实际为 %r" % (total,)
        )
    return total


def _reject_text(seq):
    if isinstance(seq, (str, bytes, bytearray)):
        raise TypeError("样本序列必须是数值列表或声道列表，不能是字符串")


def _is_nested(seq):
    """嵌套（多声道）序列：首元素是另一个序列而非数值。"""
    if not seq:
        return False
    first = seq[0]
    if isinstance(first, bool) or isinstance(first, (int, float)):
        return False
    if isinstance(first, (str, bytes, bytearray)):
        raise TypeError("声道必须是数值序列，不能是字符串")
    if not hasattr(first, "__len__"):
        raise TypeError("不支持的样本类型: %r" % (type(first).__name__,))
    return True


def _split_channels(samples):
    """返回 (声道列表, 是否多声道)，并做结构校验。"""
    _reject_text(samples)
    if not hasattr(samples, "__len__"):
        raise TypeError("样本必须是序列，实际为 %r" % (type(samples).__name__,))
    if _is_nested(samples):
        channels = list(samples)
        n = len(channels[0])
        for ch in channels[1:]:
            if len(ch) != n:
                raise LengthMismatchError(
                    "各声道长度不一致: %d != %d" % (len(ch), n)
                )
        return [list(ch) for ch in channels], True
    return list(samples), False


# ----------------------------------------------------------------------
# 纯 Python 实现（同时是 numpy 缺失时的兜底路径，单测直接覆盖）
# ----------------------------------------------------------------------

def _py_fade(flat, fade_in):
    n = len(flat)
    if n == 0:
        return []
    out = [0.0] * n
    isfinite = math.isfinite
    if n == 1:
        x = flat[0]
        if not isfinite(x):
            raise NonFiniteSampleError("样本含非有限值")
        # 单样本即末端点：淡入末端=原值，淡出末端=0
        return [x if fade_in else 0.0]
    if fade_in:
        gain = 0.0
        step = 1.0 / (n - 1)
        for i in range(n):
            x = flat[i]
            if not isfinite(x):
                raise NonFiniteSampleError("样本含非有限值")
            out[i] = x * gain
            gain += step
        out[-1] = flat[-1]  # 末端点严格等于原值，消除步长累加漂移
    else:
        gain = 1.0
        step = -1.0 / (n - 1)
        for i in range(n):
            x = flat[i]
            if not isfinite(x):
                raise NonFiniteSampleError("样本含非有限值")
            out[i] = x * gain
            gain += step
        out[0] = flat[0]  # 首端点严格等于原值
        out[-1] = 0.0  # 末端点严格为 0
    return out


def _py_cross(a, b):
    n = len(a)
    if n == 0:
        return []
    isfinite = math.isfinite
    out = [0.0] * n
    if n == 1:
        if not (isfinite(a[0]) and isfinite(b[0])):
            raise NonFiniteSampleError("样本含非有限值")
        return [a[0]]
    # 小长度直接逐点三角函数，保证端点精确；大长度用旋转递推保证速度。
    if n <= 64:
        for i in range(n):
            x, y = a[i], b[i]
            if not (isfinite(x) and isfinite(y)):
                raise NonFiniteSampleError("样本含非有限值")
            theta = _HALF_PI * i / (n - 1)
            out[i] = x * math.cos(theta) + y * math.sin(theta)
        out[0] = a[0]
        out[n - 1] = b[n - 1]
        return out
    for x in (a[0], b[0], a[n - 1], b[n - 1]):
        if not isfinite(x):
            raise NonFiniteSampleError("样本含非有限值")
    out[0] = a[0]
    out[n - 1] = b[n - 1]
    theta_step = math.pi / (2.0 * (n - 1))
    c, s = math.cos(theta_step), math.sin(theta_step)
    ga, gb = c, s  # i=1 处的增益（i=0 为 (1, 0)）
    for i in range(1, n - 1):
        x, y = a[i], b[i]
        if not (isfinite(x) and isfinite(y)):
            raise NonFiniteSampleError("样本含非有限值")
        out[i] = x * ga + y * gb
        ga, gb = ga * c - gb * s, ga * s + gb * c
    return out


# ----------------------------------------------------------------------
# numpy 加速热路径
# ----------------------------------------------------------------------

def _np_fade(flat, fade_in):
    arr = _np.asarray(flat, dtype=_np.float64)
    if not _np.isfinite(arr).all():
        raise NonFiniteSampleError("样本含非有限值")
    n = arr.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [float(arr[0]) if fade_in else 0.0]
    gains = _np.arange(n, dtype=_np.float64) / (n - 1)
    if not fade_in:
        gains = 1.0 - gains
    return (arr * gains).tolist()


def _np_cross(a, b):
    aa = _np.asarray(a, dtype=_np.float64)
    bb = _np.asarray(b, dtype=_np.float64)
    if not (_np.isfinite(aa).all() and _np.isfinite(bb).all()):
        raise NonFiniteSampleError("样本含非有限值")
    n = aa.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [float(aa[0])]
    t = _np.arange(n, dtype=_np.float64) / (n - 1)
    theta = t * _HALF_PI
    out = aa * _np.cos(theta) + bb * _np.sin(theta)
    # 端点严格对齐，消除浮点误差
    out[0] = aa[0]
    out[n - 1] = bb[n - 1]
    return out.tolist()


def _fade(samples, fade_in):
    channels, nested = _split_channels(samples)
    if nested:
        if _HAVE_NUMPY:
            stacked = _np.asarray(channels, dtype=_np.float64)
            if not _np.isfinite(stacked).all():
                raise NonFiniteSampleError("样本含非有限值")
            n = stacked.shape[1]
            if n == 0:
                return [[] for _ in channels]
            if n == 1:
                gains = _np.ones(1) if fade_in else _np.zeros(1)
            else:
                gains = _np.arange(n, dtype=_np.float64) / (n - 1)
                if not fade_in:
                    gains = 1.0 - gains
            return (stacked * gains).tolist()
        return [_py_fade(ch, fade_in) for ch in channels]
    if _HAVE_NUMPY:
        return _np_fade(channels, fade_in)
    return _py_fade(channels, fade_in)


def _cross(a, b):
    ca, na = _split_channels(a)
    cb, nb = _split_channels(b)
    if na != nb:
        raise LengthMismatchError("两端声道结构不一致（单声道/多声道混用）")
    if na:
        if len(ca) != len(cb):
            raise LengthMismatchError(
                "声道数不一致: %d != %d" % (len(ca), len(cb))
            )
        n = len(ca[0])
        for cha, chb in zip(ca, cb):
            if len(cha) != n or len(chb) != n:
                raise LengthMismatchError("交叉淡化两端长度不一致")
        if _HAVE_NUMPY:
            aa = _np.asarray(ca, dtype=_np.float64)
            bb = _np.asarray(cb, dtype=_np.float64)
            if not (_np.isfinite(aa).all() and _np.isfinite(bb).all()):
                raise NonFiniteSampleError("样本含非有限值")
            if n == 0:
                return [[] for _ in ca]
            if n == 1:
                return aa[:, :1].tolist()
            t = _np.arange(n, dtype=_np.float64) / (n - 1)
            theta = t * _HALF_PI
            out = aa * _np.cos(theta) + bb * _np.sin(theta)
            out[:, 0] = aa[:, 0]
            out[:, n - 1] = bb[:, n - 1]
            return out.tolist()
        return [_py_cross(x, y) for x, y in zip(ca, cb)]
    if len(ca) != len(cb):
        raise LengthMismatchError(
            "交叉淡化两端长度不一致: %d != %d" % (len(ca), len(cb))
        )
    if _HAVE_NUMPY:
        return _np_cross(ca, cb)
    return _py_cross(ca, cb)


class Fader:
    """无状态一次性淡化器，方法入参与返回均为 Python ``list``。

    实例不持有任何可变数据，多线程并发处理各自的缓冲区是安全的。
    """

    def fade_in(self, samples):
        """线性淡入：首样本乘 0、末样本保持原值，返回新列表。"""
        return _fade(samples, True)
    def fade_out(self, samples):
        """线性淡出：首样本保持原值、末样本乘 0，返回新列表。"""
        return _fade(samples, False)

    def crossfade(self, a, b):
        """等功率交叉淡化，要求两端等长，返回新列表。"""
        return _cross(a, b)


# ----------------------------------------------------------------------
# 流式分块处理：常数级额外内存，状态仅为当前位置与旋转增益
# ----------------------------------------------------------------------

class _BaseStream:
    __slots__ = ("total", "pos")

    def __init__(self, total):
        _validate_total(total)
        self.total = total
        self.pos = 0

    def _consume(self, length):
        if self.pos + length > self.total:
            raise LengthMismatchError(
                "流式样本数超出声明总长: 已处理 %d + 本块 %d > total %d"
                % (self.pos, length, self.total)
            )
        start = self.pos
        self.pos += length
        return start

    def flush(self):
        """声明流结束。累计长度不足时抛 :class:`LengthMismatchError`。"""
        if self.pos != self.total:
            raise LengthMismatchError(
                "流式样本数不足: 已处理 %d != total %d"
                % (self.pos, self.total)
            )
        return []

    @staticmethod
    def _check_chunk_flat(flat):
        isfinite = math.isfinite
        for x in flat:
            if not isfinite(x):
                raise NonFiniteSampleError("样本含非有限值")


class FadeInStream(_BaseStream):
    """流式线性淡入。``process`` 逐块返回结果，块间保持增益连续。"""

    __slots__ = ("total", "pos")

    def __init__(self, total):
        super().__init__(total)

    def process(self, chunk):
        channels, nested = _split_channels(chunk)
        start = self._consume(len(channels[0]) if nested else len(channels))
        if nested:
            return [self._apply(ch, start) for ch in channels]
        return self._apply(channels, start)

    def _apply(self, flat, start):
        self._check_chunk_flat(flat)
        total = self.total
        if total <= 1:
            return list(flat)  # 唯一样本即末端点，保持原值
        out = [0.0] * len(flat)
        denom = total - 1
        for j, x in enumerate(flat):
            gain = (start + j) / denom  # 按绝对帧位置计算，无累加漂移
            out[j] = x * gain
        if start + len(flat) == total:
            out[-1] = flat[-1]  # 末端点严格等于原值
        return out


class FadeOutStream(_BaseStream):
    """流式线性淡出。``process`` 逐块返回结果，块间保持增益连续。"""

    __slots__ = ("total", "pos")

    def __init__(self, total):
        super().__init__(total)

    def process(self, chunk):
        channels, nested = _split_channels(chunk)
        start = self._consume(len(channels[0]) if nested else len(channels))
        if nested:
            return [self._apply(ch, start) for ch in channels]
        return self._apply(channels, start)

    def _apply(self, flat, start):
        self._check_chunk_flat(flat)
        total = self.total
        if total <= 1:
            return [0.0] * len(flat)  # 唯一样本即末端点，增益 0
        out = [0.0] * len(flat)
        denom = total - 1
        for j, x in enumerate(flat):
            gain = 1.0 - (start + j) / denom
            out[j] = x * gain
        if start == 0 and flat:
            out[0] = flat[0]  # 首端点严格等于原值
        if start + len(flat) == total:
            out[-1] = 0.0  # 末端点严格为 0
        return out


class CrossfadeStream(_BaseStream):
    """流式等功率交叉淡化，增益按绝对帧位置计算，跨块/跨声道连续。"""

    __slots__ = ("total", "pos")

    def __init__(self, total):
        super().__init__(total)

    def process(self, chunk_a, chunk_b):
        ca, na = _split_channels(chunk_a)
        cb, nb = _split_channels(chunk_b)
        if na != nb:
            raise LengthMismatchError("两端声道结构不一致（单声道/多声道混用）")
        if na:
            if len(ca) != len(cb):
                raise LengthMismatchError(
                    "声道数不一致: %d != %d" % (len(ca), len(cb))
                )
            length = len(ca[0])
            for cha, chb in zip(ca, cb):
                if len(cha) != length or len(chb) != length:
                    raise LengthMismatchError("交叉淡化两端块长度不一致")
            start = self._consume(length)
            return [self._apply(x, y, start) for x, y in zip(ca, cb)]
        if len(ca) != len(cb):
            raise LengthMismatchError(
                "交叉淡化两端块长度不一致: %d != %d" % (len(ca), len(cb))
            )
        start = self._consume(len(ca))
        return self._apply(ca, cb, start)

    def _apply(self, a, b, start):
        isfinite = math.isfinite
        for x, y in zip(a, b):
            if not (isfinite(x) and isfinite(y)):
                raise NonFiniteSampleError("样本含非有限值")
        total = self.total
        n = len(a)
        if total <= 1:
            return list(a)  # 唯一样本对齐 a 端
        out = [0.0] * n
        denom = total - 1
        for j in range(n):
            theta = _HALF_PI * (start + j) / denom
            out[j] = a[j] * math.cos(theta) + b[j] * math.sin(theta)
        if start == 0 and n:
            out[0] = a[0]  # 首端点严格等于 a
        if start + n == total and n:
            out[n - 1] = b[n - 1]  # 末端点严格等于 b
        return out
