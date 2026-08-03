# Part2 Camera 预调用使用决策

`parking_slot_part2.camera_observability.camera_observability_tool` 是调用 Camera
工具之前的几何与遮挡判定器。它只回答一个问题：**当前场景是否值得继续调用
Camera？**

它的终态只有：

- `use_camera`
- `do_not_use_camera`
- `insufficient_information`

这个工具不读取 Camera 图像，不运行目标检测或语义分割，也不判断目标车位是
`free` 还是 `occupied`。`use_camera` 只表示几何位置、视场角和地图遮挡条件允许
后续 Camera 调用；它不是车位状态预测，也不保证后续图像一定足以得出终态。

## 生产输入合同

默认策略只接受完整上游合同：

- `target_slot_id`、`map_coordinate_frame`、`map_units_per_meter`；
- `slots` 中目标及其他车位的 `polygon_map`、Part1 状态，以及目标的
  `target_ground_z_m`（也可放在目标记录的 `ground_z_m`/`floor_z_m`）；
- 严格 `camera_calibration`：model、图像尺寸、`K`、`D`、唯一方向的
  `T_vehicle_camera`/`T_camera_vehicle`、标定 ID/时间、帧名和米制外参单位；
- `camera_map_pose`：完整 `T_map_camera`、与矩阵一致的 position/quaternion、协方差、
  localization status、时间差、标定 ID、帧名和单位；
- `static_obstacle_map`：坐标系、比例尺、带高度的墙柱 polygon，以及显式
  `mapped_regions` 完整性声明；
- 名义水平 FOV。JSON 使用 `nominal_horizontal_fov_deg`，策略使用半角。

生产默认不会把旧的 `camera_pose_map_xyyaw + calibration_id + static_obstacles: []`
解释成完整场景。旧 2D 接口仅为既有几何回归保留，必须显式设置
`allow_legacy_2d_test_contract=true`；该开关不得出现在生产配置中。

完整坐标约定和上游 schema 见 `docs/camera_observability_upstream.md`。

### 空障碍物列表不等于完整空地图

生产路径只有在目标 Camera--车位视线走廊完全落入
`static_obstacle_layer_complete=true` 的 mapped region 时，才可以把局部查询的空结果
解释为“已知没有墙柱”。缺层返回 `static_obstacle_layer_missing`；局部区域未完成建图
返回 `static_obstacle_region_incomplete`。地图其他无关区域不完整不会否决当前目标。

不得为了让判定通过而补 `static_obstacles: []`、扩大 mapped region，或把停车车辆永久
写进静态地图。

## 真实 Part1 数据接入

当前主线的车位几何来自
`outputs/full_icpark_allframes_vehicle_cluster/slot_database.json`，状态和证据来自
`outputs/slot_hybrid_3d_pose_corrected_v2_replay/slot_decisions.json`，范围信息可来自同一
运行的 `known_slot_scope.csv`。接入时必须按 `slot_id` 做 key join，生成每个车位唯一的
合并记录；不能按数组位置、CSV 行号或相邻顺序 zip。数据库中存在但没有对应 Part1
决策的车位不得推断为 `free`，应保留为未知或范围外先验。重复 ID、冲突状态或缺失目标
几何不能通过任取一条记录解决，应使本次判定 fail closed。

JSON 入口可直接接收 `slot_database`、`slot_decisions` 和 `known_slot_scope` 三个
容器，并在内部按 ID 将 decision/scope 字段叠加到对应几何记录；生产调用方仍应在
入口前验证 ID 唯一性和跨产物
hash/版本一致性，不能依赖文件顺序。

Part1 中的 `occupied_evidence.strength` 和 `free_evidence.strength` 是规则 gate 的证据
强度/排序分数，**不是校准概率**。不能把 `strength=0.8` 写成
`occupied_probability=0.8`，也不能将两个 strength 归一化成 free/occupied 概率。
`occupied_probability` 只有在独立概率模型完成校准并明确输出该字段时才可参与概率门槛。

代码已经补齐严格静态地图读取、局部完整性查询、标定读取、SE(3) 插值/组合和投影
接口；真实主线数据仍存在以下缺口：完整墙柱覆盖声明、可信 Camera 外参方向/帧名、
完整 6DoF vehicle pose/covariance/status、鱼眼模型与 D、同步元数据和实测
angle-quality 曲线。因此当前真实数据仍会 fail closed，接口完成不等于真实标定完成。

因此，直接把现有 Part1 文件拼成一个“看起来完整”的场景并不自动允许
`use_camera`。缺少关键 Camera pose、标定、目标几何或遮挡层时必须返回
`insufficient_information`。

## FOV 与角度质量

目标多边形不只用中心点判断。工具对目标中心、所有角点以及每条边的中点采样；四边形
通常得到 9 个样本。每个样本都计算相对 Camera 的 bearing，并汇总：

- `nominal_fov_coverage`：落在名义 FOV 内的样本比例；
- `reliable_fov_coverage`：达到可靠角度质量的样本比例；
- `edge_quality_score`：各样本边缘质量的聚合分数。

生产路径会在每个二维样本上按配置增加多个目标高度，并通过完整
`T_map_camera` 和 fisheye/pinhole 模型投影；默认四边形形成 `9 × 3 = 27` 个 3D 样本。
这三个 FOV 指标的分母都是全部 3D 目标样本，名义 FOV 或图像边界外的样本对质量
分数贡献为零。
`clear/blocked/uncertain` 的分母则只包含位于名义 FOV 内、且质量高于不可靠门槛的
LOS eligible 样本。

经过真实相机标定的 `angle_to_quality_curve` 优先于默认角度分区。曲线应描述绝对离轴角
与可用质量之间的关系，并且必须完整覆盖 `0°` 到 `nominal_half_fov_deg`：首个点
必须是 `0°`，末个点必须是名义半视场角，中间角度严格递增。名义 FOV 仍是硬边界。
只有没有标定曲线时，才使用
`reliable_half_fov_deg`、`edge_unreliable_start_deg` 和 `nominal_half_fov_deg` 组成的
默认中心/过渡/不可靠边缘分区。默认分区只是保守占位策略，不是传感器规格，也不能被
描述为经过数据集准确率验证。

## 三态 LOS 与 first-hit

从完整 `T_map_camera` 的 Camera 光心分别向目标中心、角点和边中点的多个高度发射
2.5D 线段。每条射线只采用目标之前
距离 Camera 最近的相交对象，即 first hit；位于目标后方的对象不遮挡目标，同一射线上
更远对象也不能覆盖更近 first hit 的语义。

每条射线恰好归入一种 LOS 状态：

- `clear`：目标之前没有可信遮挡物；
- `blocked`：first hit 是明确的 occupied 车辆占用轮廓、墙、柱或其他显式静态障碍物；
- `uncertain`：first hit 是 `unknown`、`partial_route`，或只达到潜在
  遮挡概率门槛的对象。

输出的 `clear_ray_ratio`、`blocked_ray_ratio` 和 `uncertain_ray_ratio` 分别是三类采样
射线占比。明确占用对象若提供 `occupancy_polygon_map`，优先用实际占用轮廓，而不是
整个车位框；概率字段存在时，分别使用
`occupied_probability_blocker_threshold` 和
`occupied_probability_potential_threshold`。Part1 strength 不得代替这些概率。
`visibility_corridor_width_m` 用于给有限宽度的视线走廊留出几何容差。

其他车位的状态规则是：

- `occupied`，或带 `occupancy_polygon_map` 的明确占用对象：`blocked`；
- `unknown`、无状态、`not_evaluated` 或 `unavailable`：`uncertain`；
- `partial`/`partial_route`：无概率时为 `uncertain`；只有此状态会按校准的
  `occupied_probability` 分为 blocker、potential 或非遮挡；
- `free`：不作为车辆遮挡物；只有范围状态而没有物理占用证据的 `out_of_route` 车位
  边界也不作为车辆；但 out-of-route 对象若带真实 vehicle/obstacle polygon，仍按物理
  blocker 处理；
- `clearance_proven=true`：只解除 `unknown`/`partial` 的 potential 状态；不能覆盖
  `occupied`、真实 `occupancy_polygon_map`、车辆、墙或柱等明确 blocker；
- 墙、柱、车辆等显式对象类型：`blocked`。

目标自身若为 `out_of_route`，直接得到 `do_not_use_camera` 和
`target_out_of_route`。其他 out-of-route 记录的范围标签本身不是遮挡证据，但其独立
vehicle/obstacle polygon 仍然是物理遮挡；这两条规则不能混淆。

局部遮挡不会自动否决 Camera：只要可靠 FOV 覆盖和 clear ray 比例达到门槛，且 blocked
ray 比例没有超过上限，仍可得到 `use_camera`。反之，未知对象挡住关键射线时不能假装
已知清晰。

## 配置优先级

`CameraObservabilityConfig` 的核心策略字段包括：

- `nominal_half_fov_deg`
- `reliable_half_fov_deg`
- `edge_unreliable_start_deg`
- `minimum_reliable_fov_coverage`
- `minimum_clear_ray_ratio`
- `maximum_blocked_ray_ratio`
- `occupied_probability_blocker_threshold`
- `occupied_probability_potential_threshold`
- `visibility_corridor_width_m`
- `maximum_camera_time_offset_ms`
- `maximum_camera_position_std_m`、`maximum_camera_yaw_std_deg`、
  `maximum_camera_orientation_std_deg`
- `pose_uncertainty_corridor_sigma`
- `target_sample_heights_m`、`vertical_occlusion_margin_m`
- 可选质量门槛和 `angle_to_quality_curve`

JSON 入口通过 `camera_observability_config` 传入这些字段；Python 入口也可直接传
`CameraObservabilityConfig`。调用方显式提供的策略配置覆盖代码默认门槛，但不能补造
缺失的场景事实。角度质量的来源
优先级是：与当前 Camera 标定绑定的实测曲线，其次是显式策略曲线，最后才是默认保守
分区。任何来源都不得放宽名义 FOV 硬边界。输入
`nominal_horizontal_fov_deg / 2` 必须与策略 `nominal_half_fov_deg` 一致，否则
fail closed，并输出 `camera_fov_policy_mismatch`。

## 决策优先级与 reason

规则按以下顺序 fail closed；前面的关键失败不会被后面的良好指标抵消：

1. 关键几何、真实 Camera pose、标定/FOV 或遮挡信息缺失：
   `insufficient_information`。常见 reason 为 `camera_pose_missing`、
   `camera_calibration_missing`、`target_geometry_missing`、
   `occlusion_information_missing`。
2. 目标完全位于名义 FOV 外：`do_not_use_camera`，reason 为
   `target_outside_fov`。
3. 目标主要落在不可靠边缘区域：`do_not_use_camera`，reason 为
   `target_in_unreliable_edge_region`。
4. 大部分采样射线被明确 occupied 车辆、墙或柱 first-hit 阻挡，或 blocked ratio 超过
   策略上限：`do_not_use_camera`，reason 为
   `explicit_line_of_sight_blockage`。
5. 没有明确 blocker，但关键射线存在 unknown/partial 等潜在遮挡：
   `insufficient_information`，reason 为 `unresolved_potential_occluder`。
6. `reliable_fov_coverage` 和 `clear_ray_ratio` 均通过门槛，且上述否决条件均不存在：
   `use_camera`。reason 通常包含 `target_inside_reliable_fov` 和
   `line_of_sight_mostly_clear`。

边缘否决还会输出 `insufficient_reliable_fov_coverage`。无法满足 clear-ray 门槛但也
没有形成明确 blocker 或 potential first hit 时，返回 `insufficient_information` 和
`insufficient_line_of_sight_clearance`。输入、配置或标定曲线无效时还可能输出
`invalid_input`、`invalid_camera_observability_config`、
`camera_quality_curve_invalid` 或 `camera_fov_policy_mismatch`；这些都是 fail-closed
诊断，不是 Camera 内容判断。

`reason_codes` 是机器可读审计原因，不是自然语言解释。`blocking_object_ids` 只列明确
blocker；`potential_occluder_ids` 只列未解决的潜在遮挡物。调用方应同时记录 decision、
reason 和各 coverage/ratio，不能只保留 `camera_usable`。

## 输出合同

输出字段为：

- `decision`
- `camera_usable`：仅在 `decision == "use_camera"` 时为 `true`
- `target_slot_id`
- `target_bearing_deg`
- `target_distance_m`
- `nominal_fov_coverage`
- `reliable_fov_coverage`
- `edge_quality_score`
- `clear_ray_ratio`
- `blocked_ray_ratio`
- `uncertain_ray_ratio`
- `blocking_object_ids`
- `potential_occluder_ids`
- `reason_codes`
- 可选 `schema_version` 和 `policy_version`

输出不包含 `observability`、`can_confirm_free`、`can_confirm_occupied`、`state` 或
`prediction`。当关键信息缺失时，bearing/distance 可为 `null`，coverage/ratio 保持
fail-closed 的零值，具体缺项由 `reason_codes` 给出。

一个成功的几何判定示例：

```json
{
  "schema_version": "camera-use-decision/2.1",
  "policy_version": "camera-use-policy/3.0",
  "decision": "use_camera",
  "camera_usable": true,
  "target_slot_id": "slot_0805",
  "target_bearing_deg": 3.2,
  "target_distance_m": 9.8,
  "nominal_fov_coverage": 1.0,
  "reliable_fov_coverage": 0.888889,
  "edge_quality_score": 0.94,
  "clear_ray_ratio": 0.888889,
  "blocked_ray_ratio": 0.111111,
  "uncertain_ray_ratio": 0.0,
  "blocking_object_ids": ["pillar_0012"],
  "potential_occluder_ids": [],
  "reason_codes": [
    "line_of_sight_mostly_clear",
    "target_inside_reliable_fov"
  ]
}
```

## 旧 2D 几何回归示例（禁止用于生产）

下面的输入只有 Camera/地图几何、Part1 车位先验和旧式空障碍层。
它不含图像路径、像素、检测框、分割结果、图像质量、OCR 或视觉状态预测字段。

```json
{
  "target_slot_id": "slot_0805",
  "camera_pose_map_xyyaw": [6.0000, 13.5000, 0.0],
  "camera_calibration": {
    "calibration_id": "front-left-calibration-v1",
    "angle_to_quality_curve": [
      [0.0, 1.0],
      [60.0, 1.0],
      [78.0, 0.0],
      [90.0, 0.0]
    ]
  },
  "nominal_horizontal_fov_deg": 180.0,
  "camera_observability_config": {
    "nominal_half_fov_deg": 90.0,
    "reliable_half_fov_deg": 60.0,
    "edge_unreliable_start_deg": 78.0,
    "allow_legacy_2d_test_contract": true
  },
  "map_units_per_meter": 0.04558499,
  "slots": [
    {
      "slot_id": "slot_0805",
      "polygon_map": [
        [6.3300, 13.4450],
        [6.5800, 13.4450],
        [6.5800, 13.5550],
        [6.3300, 13.5550]
      ],
      "center_map": [6.4550, 13.5000],
      "heading_deg": 0.0,
      "state": "unknown"
    }
  ],
  "static_obstacles": []
}
```

该例只用于回归旧接口。生产输入必须删除测试开关并提供严格 Camera pose、标定和带
mapped-region 完整性的静态地图；照抄本例会违反 fail-closed 合同。
