"""Deterministic conflict grouping for validated Part2 queue items."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import QueueEnvelope, QueueItem
from .queueing import canonical_sha256


GROUPING_POLICY_VERSION = "part2-conflict-groups/1.0"
MAX_GROUP_SIZE = 4


@dataclass(frozen=True, slots=True)
class TaskGroup:
    """A stable, encounter-local set of at most four decision tasks."""

    group_id: str
    encounter_id: str
    task_ids: tuple[str, ...]
    grouping_policy_version: str = GROUPING_POLICY_VERSION


def _has_edge(first: QueueItem, second: QueueItem) -> bool:
    if first.encounter.encounter_id != second.encounter.encounter_id:
        return False
    conflicts = (
        second.slot_id in first.relationships.conflict_slot_ids
        or first.slot_id in second.relationships.conflict_slot_ids
    )
    shared = bool(
        set(first.relationships.shared_evidence_ids)
        & set(second.relationships.shared_evidence_ids)
    )
    return conflicts or shared


def _connected_components(
    task_ids: tuple[str, ...],
    graph: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for start in task_ids:
        if start in seen:
            continue
        pending = [start]
        component: set[str] = set()
        while pending:
            task_id = pending.pop()
            if task_id in seen:
                continue
            seen.add(task_id)
            component.add(task_id)
            pending.extend(sorted(graph[task_id] - seen, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _split_component(
    component: tuple[str, ...],
    edges: set[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    if len(component) <= MAX_GROUP_SIZE:
        return (component,)

    component_nodes = set(component)
    uncovered = {
        edge for edge in edges if edge[0] in component_nodes and edge[1] in component_nodes
    }
    groups: list[tuple[str, ...]] = []
    grouped_nodes: set[str] = set()

    while uncovered:
        seed = min(uncovered)
        group = {seed[0], seed[1]}
        while len(group) < MAX_GROUP_SIZE:
            candidates: list[tuple[int, str]] = []
            for task_id in component_nodes - group:
                incident_count = sum(
                    1
                    for first, second in uncovered
                    if (first == task_id and second in group)
                    or (second == task_id and first in group)
                )
                if incident_count:
                    candidates.append((-incident_count, task_id))
            if not candidates:
                break
            group.add(min(candidates)[1])

        group_ids = tuple(sorted(group))
        groups.append(group_ids)
        grouped_nodes.update(group)
        uncovered = {
            edge for edge in uncovered if not (edge[0] in group and edge[1] in group)
        }

    groups.extend((task_id,) for task_id in sorted(component_nodes - grouped_nodes))
    return tuple(groups)


def _make_group(queue_id: str, encounter_id: str, task_ids: tuple[str, ...]) -> TaskGroup:
    stable_task_ids = tuple(sorted(task_ids))
    return TaskGroup(
        group_id=canonical_sha256(
            {
                "queue_id": queue_id,
                "grouping_policy_version": GROUPING_POLICY_VERSION,
                "task_ids": stable_task_ids,
            }
        ),
        encounter_id=encounter_id,
        task_ids=stable_task_ids,
    )


def build_groups(queue: QueueEnvelope) -> tuple[TaskGroup, ...]:
    """Build stable encounter-local groups from conflict/shared-evidence edges.

    Plain slot adjacency is deliberately ignored by the grouping graph.
    Components larger than four are converted to a deterministic overlapping
    edge cover so every real relationship remains reviewable.
    """

    by_encounter: dict[str, list[QueueItem]] = {}
    for item in queue.items:
        by_encounter.setdefault(item.encounter.encounter_id, []).append(item)

    groups: list[TaskGroup] = []
    for encounter_id in sorted(by_encounter):
        items = sorted(by_encounter[encounter_id], key=lambda item: item.task_id)
        task_ids = tuple(item.task_id for item in items)
        graph = {task_id: set() for task_id in task_ids}
        edges: set[tuple[str, str]] = set()
        for index, first in enumerate(items):
            for second in items[index + 1 :]:
                if not _has_edge(first, second):
                    continue
                edge = (first.task_id, second.task_id)
                graph[first.task_id].add(second.task_id)
                graph[second.task_id].add(first.task_id)
                edges.add(edge)

        for component in _connected_components(task_ids, graph):
            for group_task_ids in _split_component(component, edges):
                groups.append(_make_group(queue.queue_id, encounter_id, group_task_ids))

    return tuple(groups)


__all__ = [
    "GROUPING_POLICY_VERSION",
    "MAX_GROUP_SIZE",
    "TaskGroup",
    "build_groups",
]
