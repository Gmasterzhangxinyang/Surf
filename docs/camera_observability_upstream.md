# Camera 可用性上游合同

本文说明 `parking_slot_part2/camera_observability.py` 新增的两个上游能力：静态墙柱
地图和外参转换后的 Camera map pose。它们只服务于调用 Camera 前的几何门控，不读取
Camera 检测结果，也不产生车位 `free/occupied` 终态。

## 1. 仓库真实数据审计

当前主线车位几何来自
`outputs/full_icpark_allframes_vehicle_cluster/slot_database.json`。文件含 1397 个车位的
`polygon_map/center_map` 和 `map_units_per_meter`，但没有稳定的 map frame ID，也没有
墙柱层完整性声明。当前比例约为 `0.045584991574 map_unit/m`，运行时仍必须从产物读取，
不能把该审计值硬编码为合同。

当前 corrected trajectory 主要提供二维 `map_x/map_y/map_yaw`。它没有完整 z/roll/pitch、
6×6 covariance 和 localization status，因而不能直接生成可信 `T_map_vehicle`。数据集中
可找到旧式 K/Tr 数值，但缺少明确的 Camera model、D 语义、图像尺寸绑定、外参方向、
帧名、单位和标定时间。`fast_lio_kitti` 中的 `fov_degree: 180` 是 LiDAR 参数，不是
Camera 标定。仓库设计文档中的 90° 也只是旧策略假设。当前没有可确认的真实 Camera
水平 FOV；工具中的 180° 只是请求的名义策略输入。

根目录 GLTF 含 wall 等语义网格，但没有证明 column/pillar 等所有静态类别已完整建图，
也没有 mapped-region 覆盖审计。因此 GLTF adapter 可提取候选结构，却默认不声明区域
完整。

结论：接口和 fail-closed 链路已经可运行，但现有真实数据不足以产生生产
`use_camera`。这不是用 placeholder 数值补齐的问题。

## 2. 坐标和变换方向

全项目统一使用：`T_A_B` 将 B 坐标系中的点转换到 A 坐标系。

```text
p_vehicle = T_vehicle_camera @ p_camera
p_map     = T_map_vehicle @ p_vehicle

T_map_camera = T_map_vehicle @ T_vehicle_camera
```

若源文件明确声明保存的是 `T_camera_vehicle`，读取器先做刚体逆变换得到
`T_vehicle_camera`。读取器不会根据 `cam_to_velo`、`velo_to_cam` 等文件名猜方向。

Camera 使用 OpenCV 光学轴约定：+x 向图像右、+y 向图像下、+z 向前。map/vehicle 的
轴名必须由标定和定位数据显式声明。外参平移固定为米；若 vehicle pose 的平移使用 CAD
map unit，组合前只把外参平移乘以 `map_units_per_meter`，旋转不缩放。

## 3. 严格 Camera 标定

`parking_slot_part2.camera_calibration` 读取以下 schema：

```json
{
  "schema_version": "camera-calibration/2.0",
  "camera_model": "fisheye|pinhole",
  "image_width": 0,
  "image_height": 0,
  "K": [],
  "D": [],
  "T_vehicle_camera": [],
  "extrinsic_translation_unit": "m",
  "calibration_id": "",
  "calibration_time": "ISO-8601",
  "vehicle_frame": "vehicle",
  "camera_frame": "camera_front",
  "is_placeholder": false
}
```

`T_vehicle_camera` 与 `T_camera_vehicle` 必须且只能提供一个。fisheye 的 D 必须是四个
OpenCV equidistant polynomial 系数；pinhole 支持 4/5/8 个 OpenCV 系数。180° 名义
Camera 在生产门控中要求 fisheye 模型，不能用 pinhole 加水平角线性映射冒充。

[camera_front.placeholder.json](../configs/camera_front.placeholder.json) 只展示字段结构。
其中每个数值均为 placeholder，`is_placeholder=true`，运行时 loader 默认拒绝。

## 4. Camera map pose

`parking_slot_part2.camera_pose` 在 Camera timestamp 上对前后
`T_map_vehicle` 做平移线性插值和 quaternion SLERP，并传播 6×6 pose covariance。输入
必须显式携带 parent/child frame、translation unit、localization status 和 caller-owned
时间/协方差门槛；不外推，不猜默认定位质量。

输出为：

```json
{
  "schema_version": "camera-map-pose/1.0",
  "timestamp_ns": 0,
  "frame_id": "camera_front",
  "parent_frame": "map",
  "T_map_camera": [],
  "position_xyz": [],
  "orientation_xyzw": [],
  "pose_covariance": [],
  "localization_status": "valid|degraded|invalid",
  "time_offset_ms": 0.0,
  "calibration_id": "",
  "translation_unit": "m|map_unit",
  "map_units_per_meter": null,
  "reason_codes": []
}
```

无 bracket pose、时间过期、frame 不一致、invalid localization、超 covariance 或
placeholder calibration 时，输出 `invalid`，并把 transform/position/orientation/covariance
置为空数组。

## 5. 静态障碍物地图

`parking_slot_part2.static_obstacle_map` 支持严格 JSON/GeoJSON，以及只提取显式结构语义
的 GLTF adapter。标准 JSON 合同为：

```json
{
  "schema_version": "static-obstacle-map/1.0",
  "coordinate_frame": "map",
  "map_units_per_meter": 1.0,
  "static_obstacle_layer_present": true,
  "static_obstacles": [
    {
      "id": "column_017",
      "type": "column",
      "polygon_xy": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
      "min_z": 0.0,
      "max_z": 2.8,
      "confidence": 0.97,
      "source": "manual|cad|lidar_mapping"
    }
  ],
  "mapped_regions": [
    {
      "polygon_xy": [[-1.0, -1.0], [2.0, -1.0], [2.0, 2.0]],
      "static_obstacle_layer_complete": true
    }
  ]
}
```

`polygon_xy` 使用 `coordinate_frame` 中的 map unit；`min_z/max_z` 使用米，但垂直原点
必须仍是同一个 map frame 的原点。离线生成器只用相对地面的高度做点过滤，发布时保留
输入点的真实 map Z，绝不把每帧地面偷偷重置为 0。Camera pose 的 Z、目标
`ground_z_m` 和障碍物高度必须由上游合同声明为同一个 map frame/垂直基准；frame 声明
不一致时以 `coordinate_frame_mismatch` fail-closed，当前数据无法证明这一点时也不能发布
complete layer。

读取器验证 finite polygon、面积、自交、重复 ID、坐标系、比例尺、高度范围、confidence
和 source，并拒绝 vehicle/car/pedestrian 等动态类型。局部查询使用覆盖完整 AABB 的网格
空间索引，不按质心近邻搜索，因此长墙不会因中心点离走廊较远而漏检。

只有 Camera--target 的有限宽视线走廊完全位于 complete mapped region 内，空查询才有
“确认局部无静态障碍”的语义。其他区域 incomplete 不影响当前目标。墙柱高度与 Camera
光心、目标多高度射线做 2.5D 相交；矮路沿可以遮住低射线，但不会自动作为完整墙体。

## 6. LiDAR 离线候选生成

`scripts/build_static_obstacle_layer_from_lidar.py` 可从已有 map-frame 多帧点云离线执行地面
去除、车位区域过滤、显式动态区域过滤、空间持久性、聚类和 footprint 生成。

持久停放车辆本身也具有“多帧持久性”，因此持久性不能证明静态。没有与精确输入
identity 绑定的可信 dynamic-filter audit 时，脚本只输出 `candidate_obstacles`，正式
`static_obstacles` 保持空。只有独立 coverage audit 和 dynamic audit 都通过，才会把
对应 `mapped_regions` 标记 complete。

输入 NPZ 中的 `points_map_xyzi` 必须已经完成全三维 LiDAR-to-map 变换；脚本不会把旧式
SE(2) XY 累积误当成完整 3D map。先运行无审计候选构建以获得 `input_identity`，再由
独立审计文件绑定同一 identity：

```bash
python3 scripts/build_static_obstacle_layer_from_lidar.py \
  --frames path/to/frames.csv \
  --slot-database path/to/slot_database.json \
  --map-frame-id your/stable/map/frame \
  --points-xy-unit map_unit \
  --points-z-unit m \
  --output outputs/static_obstacle_mapping/candidates.json
```

只有第二次运行同时传入 `--dynamic-filter-audit` 和 `--coverage-audit`，且两份审计都与
上述精确 `input_identity` 匹配时，才可能发布 complete region。停放车辆即使多帧稳定，
也不会仅凭 persistence 被写入正式静态层。

## 7. Part1/地图接入新增字段

现有 `polygon_map/center_map/map_units_per_meter` 继续复用，不复制一套车位几何。接入方
需要补充或显式传入：

- `slot_database.map_frame_id`（或统一的 `coordinate_frame`）；
- 目标车位的 `ground_z_m`，与 Camera pose/静态地图共享 map-Z 基准；
- 如已有真实车辆/障碍物 footprint，使用 `occupancy_polygon_map`；没有时 occupied 车位
  才保守退化为 slot polygon；
- 独立地图资源 `static_obstacle_map`：`static_obstacle_layer_present`、
  `static_obstacles[]`、`mapped_regions[]`；
- 独立标定资源 `camera_calibration` 和逐帧 `camera_map_pose`。

车位 `state/occupied_probability` 仍来自 Part1；墙柱层不是车位状态，Camera 外参也不由
Part1 猜测。当前仓库的 Part1 产物没有稳定 frame ID 和 target ground Z，因此不能直接
满足这份生产合同。

## 8. Camera 可用性接入

生产 `camera_observability_tool` 默认要求上述三份严格合同。完整 pose 的光学中心、高度、
旋转、协方差和时差参与以下步骤：

1. 目标中心、角点、边中点按多个高度生成 3D 样本；
2. 用完整 `T_map_camera` 的逆变换和标定 model 投影到图像，检查名义 FOV 与图像边界；
3. 用标定 angle-quality curve 或保守中心/过渡/边缘策略计算质量；
4. 用位置和 yaw uncertainty 扩张局部视线走廊；
5. 对 occupied/unknown/partial 物理对象和带高度墙柱执行 first-hit；
6. 汇总 clear/blocked/uncertain 比例，只输出 Camera 使用决策。

新增 fail-closed reason 包括：

- `static_obstacle_layer_missing`
- `static_obstacle_region_incomplete`
- `camera_intrinsics_missing`
- `camera_extrinsics_missing`
- `camera_pose_missing`
- `camera_pose_stale`
- `camera_pose_uncertainty_too_high`
- `coordinate_frame_mismatch`
- `static_obstacle_blocks_view`

## 9. 验证与可视化

单元测试：

```bash
python3 -m unittest \
  tests.test_part2_camera_calibration \
  tests.test_part2_camera_pose \
  tests.test_part2_static_obstacle_map \
  tests.test_build_static_obstacle_layer_from_lidar \
  tests.test_part2_camera_observability \
  tests.test_part2_camera_observability_upstream \
  tests.test_visualize_camera_observability -v
```

Camera/LiDAR/车位/墙柱/FOV/射线投影调试：

```bash
python3 scripts/visualize_camera_observability.py \
  --scene path/to/validated_scene.json \
  --output-dir outputs/camera_observability_validation
```

scene 除严格 Camera/地图合同外还应提供 `camera_image_path`。可选
`lidar_projection_input` 必须显式声明 `coordinate_frame`、`xy_unit=map_unit`、
`z_unit=m`；可选 `reprojection_references[]` 包含 map 3D 点和人工/标定观测像素。输出为
`outputs/camera_observability_validation/<scene-stem>.png` 和同名 JSON。JSON 分别报告
left edge、center、right edge 的重投影 count/mean/RMSE/max；没有对应 reference 时
count 为 0、误差为 null，不生成虚假平均误差。图中的射线直接来自同一次
`assess_camera_observability(..., debug_trace=...)`，不是另一套几何判断。

### 独立 Part1 + Part2 map 组合图

Part1 的 `local_map.png` 合同保持不变：它只展示局部 LiDAR 覆盖、三态和 provisional
A/B，不包含 Part2 Camera pose、FOV、遮挡射线或 Camera 使用决策。独立组合图由以下
命令生成：

```bash
python3 scripts/render_part1_camera_observability_map.py \
  --local-map outputs/part1_local_lidar_frame_9277/local_map.json \
  --slot-database outputs/full_icpark_allframes_vehicle_cluster/slot_database.json
```

默认输出为：

```text
outputs/part1_local_lidar_frame_9277/local_map_camera_observability.png
outputs/part1_local_lidar_frame_9277/camera_observability_map.json
```

脚本对 Part1 最多两个 provisional A/B 目标逐一调用现有
`assess_camera_observability(..., debug_trace=...)`，不复制第二套几何判断，也不调用
Camera 语义检测模型。JSON 保留每个目标的 `assessment`、`debug_trace`、
`input_limitations`、`camera_geometry_drawn` 和所用算法/政策版本。

`--camera-scene` 是可选的严格上游 scene。未提供或内容不完整时，算法仍真实执行并返回
`insufficient_information`；渲染器不会拿 Part1 anchor/ego pose 冒充 Camera pose。
只要 Camera pose、标定、坐标 frame 或 FOV 的关键验证失败，
`camera_geometry_drawn=false`，图中就不画 Camera FOV 和射线。可用
`--target-slot-id` 覆盖 A/B 展示目标，但该参数最多重复两次，且目标必须位于当前局部
Part1 map；它不是最终停车候选选择。

frame 9277 当前缺严格 Camera map pose、确认的 180° fisheye 标定、稳定 map frame ID、
target ground-Z 和局部完整墙柱层。因此当前组合图只能诚实展示 fail-closed 结果，不能用
placeholder 或推测外参生成看似真实的 FOV/遮挡关系。

## 10. 当前真实数据限制

- 旧式 Camera 文件缺 model、D 语义、图像尺寸绑定、外参方向/单位、frame 名和标定时间；
- 没有经过确认的 180° fisheye 内参与 angle-to-quality 标定曲线；
- corrected trajectory 缺完整 z/roll/pitch、6×6 covariance 和 localization status；
- Part1 slot map 缺稳定 frame ID 和 target ground Z；
- GLTF 只能证明存在部分 wall mesh，不能证明墙/柱全覆盖，也无 mapped-region 完整性审计；
- 当前 map 点云链路尚未证明 Z 已在统一 map 垂直基准完成全三维变换；
- 没有与精确点云输入绑定的动态物体过滤和覆盖完整性审计。

因此仓库当前可以运行接口、校验、测试和可视化工具，但真实场景仍会 fail-closed，不能
声称已经得到生产 `use_camera`。
