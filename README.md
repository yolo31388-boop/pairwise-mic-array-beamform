# pairwise-mic-array-beamform

## 领域

游戏、图形与多媒体

## 主题

麦克风阵列波束成形

## 说明

实测几何、双讲冻结与采样对齐。

初始状态下 `media/mic_array_beamform.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
