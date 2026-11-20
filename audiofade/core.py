"""audiofade 核心 API（初始未实现）。"""


class Fader:
    def fade_in(self, samples):
        """淡入"""
        raise NotImplementedError

    def fade_out(self, samples):
        """淡出"""
        raise NotImplementedError

    def crossfade(self, a, b):
        """交叉淡化"""
        raise NotImplementedError
