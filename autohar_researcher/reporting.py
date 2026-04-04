from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from autohar_researcher.types import CandidateResult


def write_history_csv(
    history: list[CandidateResult],
    output_path: Path,
    *,
    history_kind: str,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if history_kind == "trial":
            writer.writerow(
                [
                    "round_id",
                    "candidate_name",
                    "branch_name",
                    "trial_val_score",
                    "accepted_in_round",
                    "best_so_far_score",
                ]
            )
        else:
            writer.writerow(
                [
                    "round_id",
                    "accepted_agent_name",
                    "accepted_branch_name",
                    "accepted_val_score",
                    "accepted",
                    "best_so_far_score",
                ]
            )
        for result in history:
            writer.writerow(
                [
                    result.round_id,
                    result.agent_name,
                    result.branch_name,
                    f"{result.val_score:.4f}",
                    str(result.accepted).lower(),
                    f"{result.best_so_far_score:.4f}",
                ]
            )
    return str(output_path)


def write_history_svg(
    history: list[CandidateResult],
    output_path: Path,
    value_key: str,
    title: str,
    *,
    legend_labels: dict[str, str] | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for result in history:
        value = result.val_score if value_key == "val_score" else result.best_so_far_score
        grouped[result.agent_name].append((result.round_id, value))

    width = 960
    height = 520
    margin_left = 70
    margin_right = 40
    margin_top = 40
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_round = max((result.round_id for result in history), default=1)
    colors = {
        "model_agent": "#0b7285",
        "augmentation_agent": "#e67700",
        "data_agent": "#2b8a3e",
        "orchestrator": "#c2255c",
    }

    def x_coord(round_id: int) -> float:
        if max_round <= 1:
            return margin_left + plot_width / 2.0
        return margin_left + ((round_id - 1) / (max_round - 1)) * plot_width

    def y_coord(score: float) -> float:
        return margin_top + (1.0 - score) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="24" font-family="Arial, sans-serif" font-size="20" fill="#111">{title}</text>',
    ]

    for tick in range(0, 6):
        score = tick / 5.0
        y = y_coord(score)
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="1"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#444">{score:.1f}</text>')

    for tick in range(1, max_round + 1):
        if max_round > 12 and tick not in {1, max_round} and tick % 5 != 0:
            continue
        x = x_coord(tick)
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#efefef" stroke-width="1"/>')
        elements.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#444">{tick}</text>')

    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5"/>')
    elements.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5"/>')
    elements.append(f'<text x="{width / 2:.1f}" y="{height - 15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111">Round</text>')
    elements.append(f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.1f})" font-family="Arial, sans-serif" font-size="14" fill="#111">Macro-F1</text>')

    legend_x = width - margin_right - 210
    legend_y = margin_top + 10
    for index, (agent_name, points) in enumerate(sorted(grouped.items())):
        color = colors.get(agent_name, "#555")
        y = legend_y + index * 22
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        label = legend_labels.get(agent_name, agent_name) if legend_labels else agent_name
        elements.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#222">{label}</text>')
        ordered = sorted(points)
        polyline = " ".join(f"{x_coord(round_id):.1f},{y_coord(score):.1f}" for round_id, score in ordered)
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
        for round_id, score in ordered:
            elements.append(
                f'<circle cx="{x_coord(round_id):.1f}" cy="{y_coord(score):.1f}" r="3.5" fill="{color}"/>'
            )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return str(output_path)


def write_promotion_svg(history: list[CandidateResult], output_path: Path, title: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(history, key=lambda result: result.round_id)

    width = 960
    height = 520
    margin_left = 70
    margin_right = 40
    margin_top = 40
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_round = max((result.round_id for result in ordered), default=1)
    color = "#c2255c"

    def x_coord(round_id: int) -> float:
        if max_round <= 1:
            return margin_left + plot_width / 2.0
        return margin_left + ((round_id - 1) / (max_round - 1)) * plot_width

    def y_coord(score: float) -> float:
        return margin_top + (1.0 - score) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="24" font-family="Arial, sans-serif" font-size="20" fill="#111">{title}</text>',
    ]

    for tick in range(0, 6):
        score = tick / 5.0
        y = y_coord(score)
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="1"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#444">{score:.1f}</text>')

    for tick in range(1, max_round + 1):
        if max_round > 12 and tick not in {1, max_round} and tick % 5 != 0:
            continue
        x = x_coord(tick)
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#efefef" stroke-width="1"/>')
        elements.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#444">{tick}</text>')

    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5"/>')
    elements.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5"/>')
    elements.append(f'<text x="{width / 2:.1f}" y="{height - 15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111">Round</text>')
    elements.append(f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.1f})" font-family="Arial, sans-serif" font-size="14" fill="#111">Macro-F1</text>')

    legend_x = width - margin_right - 260
    legend_y = margin_top + 10
    elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
    elements.append(f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#222">accepted_baseline</text>')
    polyline = " ".join(f"{x_coord(result.round_id):.1f},{y_coord(result.val_score):.1f}" for result in ordered)
    if polyline:
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
    for result in ordered:
        x = x_coord(result.round_id)
        y = y_coord(result.val_score)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        elements.append(
            f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#444">{result.agent_name}</text>'
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return str(output_path)
