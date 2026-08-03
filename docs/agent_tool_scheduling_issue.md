# ParkingAgent v2 工具调度一致性问题

## 问题

ParkingAgent v2 曾同时使用三处规则决定工具是否合法：

1. 发给模型的 `available_tools`；
2. 系统提示词中的工具策略；
3. Python 运行时的动作校验。

三处规则不一致会让 Agent 看到一个工具可用，但执行时被拒绝；也会让
提示词允许重新验证定位假设，而 `available_tools` 不再提供对应工具。

### 问题一：Camera Sequence 过早暴露

旧实现会在尚未成功执行 `camera_context` 时把 `camera_sequence` 放进
`available_tools`，但运行时又要求 Sequence 必须先有成功的 Context。
Agent 选择该工具后只能收到 `ActionError`，浪费一次模型响应。

Context 执行失败时也不能仅凭“调用过 Context”开放 Crop 或 Sequence；
依赖工具必须以 `camera_context.status == "ok"` 为前提。

### 问题二：Camera Crop 重试合同冲突

提示词允许第一次 Crop 反驳或无法确认定位假设后，修改 bbox 或增强方式
再次 Crop。旧调度器只在从未调用过 Crop 时展示 `camera_crop`，因此遵守
`available_tools` 的 Agent 无法重试。

底层执行器却没有禁止第二次 Crop，也没有拒绝完全相同的 bbox 和增强参数。
这会产生两个相反行为：

- 遵守菜单的 Agent 无法修正定位假设；
- 忽略菜单的 Agent 可能重复执行完全相同、没有信息增益的 Crop。

## 修复后的状态机

`available_tools` 是合法动作的唯一事实来源，运行时在执行前再次检查同一
结果。

| 当前状态 | 可用工具 |
|---|---|
| FOV `not_visible` | 尚未尝试且具备增量证据时仅 `lidar_detail` |
| Camera 合法、Context 未尝试 | `camera_context`；有增量证据时另有 `lidar_detail` |
| Context 失败或不可用 | 不开放 Crop/Sequence；仍可使用合法的 `lidar_detail` |
| Context 成功 | 未尝试的 `camera_sequence`、`camera_crop`、合法的 `lidar_detail` |
| Crop 已执行且仍有轮次 | 继续开放 `camera_crop`，允许修正假设 |

Crop 重试必须满足：

- 新 bbox 与历史调用不同；或
- 新 enhancement 与历史调用不同。

完全相同的 `(bbox_norm, enhancement)` 组合会被拒绝，不消耗证据工具轮次，
错误反馈给模型重新规划。

## 三轮预算示例

Camera 合法时常见的有效路线为：

```text
camera_context -> camera_crop -> lidar_detail
```

如果第一次 Crop 明确选错区域，也允许：

```text
camera_context -> camera_crop(H1) -> camera_crop(H2)
```

第二条路线会用完三轮预算，因此只适合完整可见且可能形成高置信
Camera-only 终态的案例；Partial/Uncertain FOV 通常应保留最后一轮给
`lidar_detail`。

## 回归验收

- Context 前不展示 Sequence 或 Crop。
- Context 失败后不展示 Sequence 或 Crop。
- Context 成功后展示 Sequence 和 Crop。
- 第一次 Crop 后仍展示 Crop。
- 改变 bbox 或 enhancement 的重试通过。
- 完全相同的 Crop 重试被拒绝。
- 模型输出不在 `available_tools` 中的工具时，运行时拒绝执行。
