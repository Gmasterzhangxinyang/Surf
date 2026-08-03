#!/usr/bin/env python3
"""Convert this KITTI-style dataset to a FAST-LIO friendly ROS1 bag.

Requires a ROS1 Python environment with rospy, rosbag and sensor_msgs.
This script does not use dataset pose files. It writes:
  /velodyne_points  sensor_msgs/PointCloud2 with x/y/z/intensity/time fields
  /imu/data         sensor_msgs/Imu from KITTI/OXTS acceleration and gyro
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
TIMESTAMP_ROOT = Path("/home/ParkingAgent/dataset/dataset/timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a ROS1 bag for FAST-LIO from KITTI-style LiDAR/OXTS files")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--timestamp-root", type=Path, default=TIMESTAMP_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/fast_lio_kitti/dataset_fastlio.bag"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means all frames from start")
    parser.add_argument("--lidar-topic", default="/velodyne_points")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--lidar-frame", default="velodyne")
    parser.add_argument("--imu-frame", default="imu")
    parser.add_argument("--scan-period", type=float, default=0.1, help="Velodyne scan period in seconds")
    parser.add_argument(
        "--imu-fields",
        choices=("body", "xyz"),
        default="body",
        help="body uses KITTI af/al/au and wf/wl/wu; xyz uses ax/ay/az and wx/wy/wz",
    )
    return parser.parse_args()


def require_ros() -> tuple[object, object, object, object, object, object]:
    try:
        import rosbag
        import rospy
        from sensor_msgs.msg import Imu, PointCloud2, PointField
        from std_msgs.msg import Header
    except Exception as exc:
        print(
            "ROS1 Python modules are not available. Run this inside a ROS1 environment "
            "with rospy, rosbag and sensor_msgs installed.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return rosbag, rospy, Header, Imu, PointCloud2, PointField


def load_timestamp_map(path: Path, tabbed: bool) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if tabbed else line.split()
        if len(parts) < 2:
            continue
        out[int(parts[0])] = float(parts[1])
    return out


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def point_times(xy: np.ndarray, scan_period: float) -> np.ndarray:
    az = np.arctan2(xy[:, 1], xy[:, 0])
    az = np.mod(az, 2.0 * math.pi)
    return (az / (2.0 * math.pi) * scan_period).astype(np.float32)


def make_cloud_msg(path: Path, stamp: object, frame_id: str, scan_period: float, Header: object, PointCloud2: object, PointField: object) -> object:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"invalid Velodyne bin: {path}")
    pts = raw.reshape(-1, 4)
    times = point_times(pts[:, :2], scan_period)
    packed = np.column_stack([pts[:, :4], times]).astype("<f4", copy=False)

    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = int(len(packed))
    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
        PointField("time", 16, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step = 20
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = packed.tobytes()
    return msg


def make_imu_msg(path: Path, stamp: object, frame_id: str, imu_fields: str, Header: object, Imu: object) -> object:
    vals = np.fromstring(path.read_text().strip(), sep=" ")
    if vals.size < 24:
        raise ValueError(f"invalid OXTS row: {path}")
    roll, pitch, yaw = vals[3], vals[4], vals[5]
    if imu_fields == "body":
        acc = vals[[15, 16, 17]]
        gyr = vals[[21, 22, 23]]
    else:
        acc = vals[[12, 13, 14]]
        gyr = vals[[18, 19, 20]]

    qx, qy, qz, qw = quat_from_rpy(float(roll), float(pitch), float(yaw))
    msg = Imu()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.orientation.x = qx
    msg.orientation.y = qy
    msg.orientation.z = qz
    msg.orientation.w = qw
    msg.angular_velocity.x = float(gyr[0])
    msg.angular_velocity.y = float(gyr[1])
    msg.angular_velocity.z = float(gyr[2])
    msg.linear_acceleration.x = float(acc[0])
    msg.linear_acceleration.y = float(acc[1])
    msg.linear_acceleration.z = float(acc[2])
    return msg


def run() -> None:
    args = parse_args()
    rosbag, rospy, Header, Imu, PointCloud2, PointField = require_ros()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lidar_ts = load_timestamp_map(args.timestamp_root / "timestamp_velodyne.txt", tabbed=True)
    imu_ts = load_timestamp_map(args.timestamp_root / "timestamps_oxts.txt", tabbed=False)
    lidar_files = sorted((args.dataset_root / "velodyne").glob("*.bin"))
    lidar_files = [p for p in lidar_files if int(p.stem) >= args.start_index]
    if args.num_frames > 0:
        lidar_files = lidar_files[: args.num_frames]

    if not lidar_files:
        raise SystemExit("no LiDAR frames selected")
    start_frame = int(lidar_files[0].stem)
    end_frame = int(lidar_files[-1].stem)
    start_t = lidar_ts[start_frame] - 0.5
    end_t = lidar_ts[end_frame] + 0.5
    imu_indices = [idx for idx, ts in imu_ts.items() if start_t <= ts <= end_t]

    with rosbag.Bag(str(args.output), "w") as bag:
        for idx in imu_indices:
            stamp = rospy.Time.from_sec(imu_ts[idx])
            msg = make_imu_msg(args.dataset_root / "oxts" / f"{idx:06d}.txt", stamp, args.imu_frame, args.imu_fields, Header, Imu)
            bag.write(args.imu_topic, msg, stamp)
        for path in lidar_files:
            idx = int(path.stem)
            stamp = rospy.Time.from_sec(lidar_ts[idx])
            msg = make_cloud_msg(path, stamp, args.lidar_frame, args.scan_period, Header, PointCloud2, PointField)
            bag.write(args.lidar_topic, msg, stamp)

    print(f"[bag] {args.output}")
    print(f"[lidar] frames={len(lidar_files)} topic={args.lidar_topic}")
    print(f"[imu] frames={len(imu_indices)} topic={args.imu_topic}")


if __name__ == "__main__":
    run()
