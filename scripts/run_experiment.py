from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "scripts"))

from client.replay_requests import load_requests
from server.config import load_all_configs
from server.entities import NetworkProfile, Request
from server.gateway import ExperimentGateway
from server.metrics import (
    build_ablation_summary,
    build_action_distribution,
    build_main_summary,
    build_privacy_summary,
    build_semantic_threshold_summary,
    build_vram_budget,
    save_rows,
    write_report,
)
from plot_results import generate_all_figures


ORDINARY_TOPICS = [
    ("edge_sched", "边缘计算中的任务卸载与带宽权衡", "时延、带宽与显存协调"),
    ("tcp_bdp", "RTT 与带宽时延积对传输效率的影响", "网络测量与调优"),
    ("quant_tradeoff", "4-bit 量化对 LLM 时延和质量的影响", "性能与精度平衡"),
    ("gpu_fragment", "GPU 显存碎片为何会影响推理稳定性", "工程原因与缓解办法"),
    ("rag_pipeline", "RAG 检索、重排和生成三阶段协同", "系统链路优化"),
    ("async_python", "Python asyncio 与线程模型的差异", "高并发服务设计"),
    ("cache_design", "语义缓存与精确缓存的区别", "命中率与错误复用风险"),
    ("qos_tail", "为什么网络系统要关注 P95/P99 时延", "尾延迟与服务质量"),
]

RAG_TOPICS = [
    ("rag_campus", "校园网络故障复盘摘要", "根因、影响范围与恢复步骤"),
    ("rag_edge_ops", "边缘节点部署记录", "模型发布、回滚和监控策略"),
    ("rag_health", "医疗随访知识摘录", "病例归档与隐私边界"),
    ("rag_finance", "财务采购制度摘录", "审批链路与预算控制"),
    ("rag_security", "安全告警调查手册", "事件分类与处置优先级"),
    ("rag_research", "科研方案会议纪要", "实验变量与对比基线"),
]

LONG_TOPICS = [
    ("long_weekly", "多周项目周报与例会纪要", "压缩为执行摘要和行动项"),
    ("long_logs", "服务端日志与监控记录", "提炼异常模式和定位线索"),
    ("long_literature", "论文阅读笔记合集", "归纳研究脉络与创新点"),
    ("long_contract", "采购与运维文档合集", "梳理风险、依赖和交付节点"),
    ("long_rag_doc", "知识库长文档", "总结主线并输出 FAQ"),
    ("long_incident", "网络故障多阶段复盘", "构建时间线与根因链路"),
]

PRIVACY_TOPICS = [
    ("privacy_student", "学生信息整理请求", "需要脱敏后给出处理建议"),
    ("privacy_customer", "客户投诉工单分类", "提炼诉求并隐藏个人信息"),
    ("privacy_employee", "员工账户异常排查", "保留问题上下文但删除标识符"),
    ("privacy_medical", "门诊记录摘要生成", "保护患者隐私并输出随访建议"),
    ("privacy_vendor", "供应商合同审批备注", "抽取关键信息并屏蔽联系方式"),
    ("privacy_repair", "网络报修单处理", "定位故障时保留必要上下文"),
]


def stable_seed(base_seed: int, *parts: object) -> int:
    value = base_seed
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2**32)
    return value


def zipf_weights(length: int, exponent: float = 0.78) -> List[float]:
    return [1.0 / ((index + 1) ** exponent) for index in range(length)]


def pick_name(rng: random.Random) -> str:
    family = ["张", "李", "王", "赵", "陈", "刘", "杨", "黄"]
    given = ["晨", "宇", "涵", "婷", "浩", "宁", "博", "悦"]
    return rng.choice(family) + rng.choice(given)


def pick_phone(rng: random.Random) -> str:
    return "1" + "".join(rng.choice("3456789")) + "".join(rng.choice("0123456789") for _ in range(9))


def pick_email(rng: random.Random, name: str) -> str:
    local = f"{name}{rng.randint(10,99)}".encode("utf-8").hex()[:8]
    return f"{local}@example.com"


def pick_address(rng: random.Random) -> str:
    city = rng.choice(["上海", "北京", "杭州", "深圳", "广州"])
    road = rng.choice(["学府路", "创新大道", "科苑路", "文汇路", "园区路"])
    return f"{city}{road}{rng.randint(10, 400)}号"


def pick_student_id(rng: random.Random) -> str:
    return "20" + "".join(rng.choice("0123456789") for _ in range(8))


def pick_account_id(rng: random.Random) -> str:
    return "ACC-" + "".join(rng.choice("0123456789ABCDEF") for _ in range(6))


def ordinary_prompt(topic: str, focus: str, variant: int) -> str:
    templates = [
        "请用结构化方式解释{topic}，重点说明{focus}。",
        "如果我要写课程报告，怎样介绍{topic}？请聚焦{focus}。",
        "围绕{topic}写一个简洁但完整的回答，强调{focus}。",
        "请比较{topic}在真实系统中的常见做法，并总结{focus}。",
    ]
    return templates[variant % len(templates)].format(topic=topic, focus=focus)


def rag_prompt(topic: str, focus: str, variant: int) -> str:
    templates = [
        "参考资料：{topic}。请基于给定材料总结{focus}，不要编造未给出的事实。",
        "以下是资料摘要：{topic}。请回答与{focus}相关的问题，并保持可核查。",
        "请阅读资料《{topic}》，输出关于{focus}的结论、证据和注意事项。",
        "资料内容围绕{topic}展开。请据此生成关于{focus}的问答式总结。",
    ]
    return templates[variant % len(templates)].format(topic=topic, focus=focus)


def long_prompt(topic: str, focus: str, variant: int) -> str:
    templates = [
        "系统提示：你需要阅读一份长文档。文档主题是{topic}，目标是{focus}。请输出执行摘要、关键证据和行动列表。",
        "请处理以下长上下文资料，主题为{topic}。任务：{focus}，并按章节组织结果。",
        "你收到一份长文档集合，核心话题是{topic}。请完成{focus}，同时指出冲突信息。",
        "以下内容来自多段项目材料，围绕{topic}。请根据这些资料完成{focus}。",
    ]
    return templates[variant % len(templates)].format(topic=topic, focus=focus)


def privacy_prompt(topic: str, focus: str, variant: int, rng: random.Random) -> str:
    name = pick_name(rng)
    phone = pick_phone(rng)
    email = pick_email(rng, name)
    address = pick_address(rng)
    student_id = pick_student_id(rng)
    account_id = pick_account_id(rng)
    templates = [
        "请处理以下请求：姓名{name}，手机号{phone}，邮箱{email}，地址{address}，学号{student_id}，账号{account_id}。主题是{topic}，目标是{focus}。",
        "有一条敏感工单需要整理：客户{name}，联系电话{phone}，联系邮箱{email}，地址{address}，业务编号{account_id}。请围绕{topic}完成{focus}。",
        "下面是一段包含个人信息的记录：姓名{name}，学号{student_id}，住址{address}，手机号{phone}。请根据{topic}输出{focus}。",
        "请阅读这条隐私相关文本：员工{name}，账号{account_id}，邮箱{email}，地址{address}。围绕{topic}完成{focus}。",
    ]
    return templates[variant % len(templates)].format(
        name=name,
        phone=phone,
        email=email,
        address=address,
        student_id=student_id,
        account_id=account_id,
        topic=topic,
        focus=focus,
    )


def make_request(
    rng: random.Random,
    request_id: str,
    cluster_id: str,
    category: str,
    topic: str,
    focus: str,
    variant_id: int,
    arrival_ms: float,
) -> Request:
    if category == "ordinary":
        prompt = ordinary_prompt(topic, focus, variant_id)
        prefix_id = "qa_system"
        prefix_tokens = rng.randint(96, 180)
        prompt_tokens = prefix_tokens + rng.randint(48, 240)
        output_tokens = rng.randint(80, 150)
        difficulty = rng.uniform(0.22, 0.64)
        sensitive = False
    elif category == "rag":
        prompt = rag_prompt(topic, focus, variant_id)
        prefix_id = f"rag_template_{(variant_id % 3) + 1}"
        prefix_tokens = rng.randint(240, 620)
        prompt_tokens = prefix_tokens + rng.randint(120, 420)
        output_tokens = rng.randint(100, 180)
        difficulty = rng.uniform(0.35, 0.78)
        sensitive = False
    elif category == "long_context":
        prompt = long_prompt(topic, focus, variant_id)
        prefix_id = f"long_doc_{(hash(cluster_id) % 3) + 1}"
        prefix_tokens = rng.randint(900, 2600)
        prompt_tokens = prefix_tokens + rng.randint(400, 2600)
        output_tokens = rng.randint(140, 240)
        difficulty = rng.uniform(0.48, 0.88)
        sensitive = False
    else:
        prompt = privacy_prompt(topic, focus, variant_id, rng)
        prefix_id = f"privacy_case_{(variant_id % 3) + 1}"
        prefix_tokens = rng.randint(140, 360)
        prompt_tokens = prefix_tokens + rng.randint(80, 260)
        output_tokens = rng.randint(70, 150)
        difficulty = rng.uniform(0.30, 0.72)
        sensitive = True

    quality_target = min(0.97, 0.70 + difficulty * 0.26 + (0.04 if category in {"rag", "long_context"} else 0.0))
    return Request(
        request_id=request_id,
        prompt=prompt,
        category=category,
        topic=topic,
        cluster_id=cluster_id,
        variant_id=variant_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefix_id=prefix_id,
        prefix_tokens=prefix_tokens,
        difficulty=round(difficulty, 4),
        quality_target=round(quality_target, 4),
        arrival_ms=round(arrival_ms, 3),
        sensitive_expected=sensitive,
    )


def build_general_clusters() -> List[Tuple[str, str, str, str]]:
    clusters: List[Tuple[str, str, str, str]] = []
    for cluster_id, topic, focus in ORDINARY_TOPICS:
        clusters.append((cluster_id, "ordinary", topic, focus))
    for cluster_id, topic, focus in RAG_TOPICS:
        clusters.append((cluster_id, "rag", topic, focus))
    for cluster_id, topic, focus in LONG_TOPICS:
        clusters.append((cluster_id, "long_context", topic, focus))
    for cluster_id, topic, focus in PRIVACY_TOPICS:
        clusters.append((cluster_id, "privacy", topic, focus))
    return clusters


def generate_general_requests(path: Path, size: int, seed: int) -> None:
    rng = random.Random(seed)
    clusters = build_general_clusters()
    weights = zipf_weights(len(clusters))
    arrival_ms = 0.0
    rows: List[Dict[str, object]] = []
    for index in range(size):
        cluster_id, category, topic, focus = rng.choices(clusters, weights=weights, k=1)[0]
        variant_id = rng.randint(0, 3)
        inter_arrival = rng.expovariate(1 / 680.0)
        if rng.random() < 0.10:
            inter_arrival = rng.uniform(60.0, 180.0)
        arrival_ms += inter_arrival
        request = make_request(
            rng=rng,
            request_id=f"main-{index:04d}",
            cluster_id=cluster_id,
            category=category,
            topic=topic,
            focus=focus,
            variant_id=variant_id,
            arrival_ms=arrival_ms,
        )
        rows.append(request.__dict__)
    write_jsonl(path, rows)


def generate_privacy_requests(path: Path, size: int, seed: int) -> None:
    rng = random.Random(seed)
    normal_clusters = [(cluster_id, "ordinary", topic, focus) for cluster_id, topic, focus in ORDINARY_TOPICS[:4]]
    sensitive_clusters = [(cluster_id, "privacy", topic, focus) for cluster_id, topic, focus in PRIVACY_TOPICS]
    rows: List[Dict[str, object]] = []
    arrival_ms = 0.0
    half = size // 2
    for index in range(half):
        cluster_id, category, topic, focus = normal_clusters[index % len(normal_clusters)]
        variant_id = rng.randint(0, 3)
        arrival_ms += rng.expovariate(1 / 560.0)
        request = make_request(
            rng=rng,
            request_id=f"privacy-normal-{index:04d}",
            cluster_id=cluster_id,
            category=category,
            topic=topic,
            focus=focus,
            variant_id=variant_id,
            arrival_ms=arrival_ms,
        )
        rows.append(request.__dict__)

    for index in range(size - half):
        cluster_id, category, topic, focus = sensitive_clusters[index % len(sensitive_clusters)]
        variant_id = rng.randint(0, 3)
        arrival_ms += rng.expovariate(1 / 560.0)
        request = make_request(
            rng=rng,
            request_id=f"privacy-sensitive-{index:04d}",
            cluster_id=cluster_id,
            category=category,
            topic=topic,
            focus=focus,
            variant_id=variant_id,
            arrival_ms=arrival_ms,
        )
        rows.append(request.__dict__)
    write_jsonl(path, rows)


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + "\n", encoding="utf-8")


def ensure_datasets(configs: Dict[str, object]) -> tuple[Path, Path]:
    experiment = configs["experiment"]
    hardware = configs["hardware"]
    prompt_path = BASE_DIR / "data" / "prompts.jsonl"
    privacy_path = BASE_DIR / "data" / "privacy_prompts.jsonl"
    generate_general_requests(prompt_path, int(experiment["dataset_size"]), seed=stable_seed(hardware["seed"], "main-dataset"))
    generate_privacy_requests(privacy_path, int(experiment["privacy_dataset_size"]), seed=stable_seed(hardware["seed"], "privacy-dataset"))
    return prompt_path, privacy_path


def load_networks(configs: Dict[str, object]) -> List[NetworkProfile]:
    return [NetworkProfile(**profile) for profile in configs["network_profiles"]["profiles"]]


def run_batch(
    configs: Dict[str, object],
    requests: List[Request],
    networks: List[NetworkProfile],
    strategies: Iterable[str],
    experiment_name: str,
    semantic_threshold: float | None = None,
) -> List[Dict[str, object]]:
    base_seed = int(configs["hardware"]["seed"])
    rows: List[Dict[str, object]] = []
    for network in networks:
        for strategy in strategies:
            gateway = ExperimentGateway(configs, seed=stable_seed(base_seed, experiment_name, network.name, strategy, semantic_threshold or "default"))
            for request in requests:
                row = gateway.handle_request(request, network, strategy, semantic_threshold=semantic_threshold)
                row["experiment_name"] = experiment_name
                rows.append(row)
    return rows


def main() -> None:
    configs = load_all_configs(BASE_DIR)
    prompt_path, privacy_path = ensure_datasets(configs)
    requests = load_requests(prompt_path)
    privacy_requests = load_requests(privacy_path)
    networks = load_networks(configs)

    experiment = configs["experiment"]
    semantic_thresholds = [float(value) for value in experiment["semantic_thresholds"]]

    main_rows = run_batch(
        configs=configs,
        requests=requests,
        networks=networks,
        strategies=experiment["strategies"],
        experiment_name="main",
    )
    main_frame = save_rows(main_rows, BASE_DIR / "results" / "raw_logs.csv")
    main_summary = build_main_summary(main_frame[main_frame["experiment_name"] == "main"])
    main_summary.to_csv(BASE_DIR / "results" / "main_summary.csv", index=False)

    privacy_rows = run_batch(
        configs=configs,
        requests=privacy_requests,
        networks=networks,
        strategies=["no_privacy", "ours"],
        experiment_name="privacy_eval",
    )
    privacy_frame = pd.DataFrame(privacy_rows)
    privacy_frame.to_csv(BASE_DIR / "results" / "privacy_raw_logs.csv", index=False)
    privacy_summary = build_privacy_summary(privacy_frame)
    privacy_summary.to_csv(BASE_DIR / "results" / "privacy_summary.csv", index=False)

    semantic_rows: List[Dict[str, object]] = []
    mobile_network = [network for network in networks if network.name == "mobile_4g5g"]
    for threshold in semantic_thresholds:
        semantic_rows.extend(
            run_batch(
                configs=configs,
                requests=requests,
                networks=mobile_network,
                strategies=["semantic_cache_only"],
                experiment_name="semantic_sweep",
                semantic_threshold=threshold,
            )
        )
    semantic_frame = pd.DataFrame(semantic_rows)
    semantic_frame.to_csv(BASE_DIR / "results" / "semantic_sweep_raw_logs.csv", index=False)
    baseline_mean = float(
        main_summary[
            (main_summary["network"] == "mobile_4g5g") & (main_summary["strategy"] == "cloud_only")
        ]["mean_e2e_ms"].iloc[0]
    )
    semantic_summary = build_semantic_threshold_summary(semantic_frame, baseline_mean_e2e=baseline_mean)
    semantic_summary.to_csv(BASE_DIR / "results" / "semantic_threshold_sweep.csv", index=False)

    ablation_rows = run_batch(
        configs=configs,
        requests=requests,
        networks=networks,
        strategies=experiment["ablation_methods"],
        experiment_name="ablation",
    )
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_frame.to_csv(BASE_DIR / "results" / "ablation_raw_logs.csv", index=False)
    ablation_summary = build_ablation_summary(ablation_frame)
    ablation_summary.to_csv(BASE_DIR / "results" / "ablation_summary.csv", index=False)

    action_distribution = build_action_distribution(main_frame[(main_frame["experiment_name"] == "main") & (main_frame["strategy"] == "ours")])
    action_distribution.to_csv(BASE_DIR / "results" / "action_distribution.csv", index=False)

    vram_budget = build_vram_budget(configs)
    vram_budget.to_csv(BASE_DIR / "results" / "vram_budget.csv", index=False)

    write_report(BASE_DIR, main_summary, privacy_summary, semantic_summary, ablation_summary)
    generate_all_figures(BASE_DIR)

    print("Experiment completed.")
    print(f"Raw logs: {BASE_DIR / 'results' / 'raw_logs.csv'}")
    print(f"Figures: {BASE_DIR / 'results' / 'figures'}")


if __name__ == "__main__":
    main()
