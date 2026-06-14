from __future__ import annotations

import math
import re
from typing import Dict, Tuple


PHONE_RE = re.compile(r"1[3-9]\d{9}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
STUDENT_ID_RE = re.compile(r"(20\d{8}|S\d{6,10}|学号[:：]?\s*\d{6,12})")
ACCOUNT_RE = re.compile(r"(账号|工号|编号|订单号|合同号|病历号)[:：]?\s*[A-Za-z0-9-]{4,}")

LOCATION_TERMS = [
    "地址",
    "住址",
    "宿舍",
    "教学楼",
    "实验室",
    "上海",
    "北京",
    "广州",
    "深圳",
    "杭州",
]
DOMAIN_TERMS = [
    "预算",
    "合同",
    "客户",
    "投标",
    "账号",
    "报销",
    "医疗",
    "诊断",
    "成绩",
    "采购",
]
NAME_TERMS = ["姓名", "同学", "老师", "张", "李", "王", "赵", "陈", "刘"]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def score_prompt(text: str) -> Tuple[float, Dict[str, int]]:
    phone_count = len(PHONE_RE.findall(text))
    email_count = len(EMAIL_RE.findall(text))
    student_count = len(STUDENT_ID_RE.findall(text))
    account_count = len(ACCOUNT_RE.findall(text))

    n_id = phone_count + student_count + account_count
    n_loc = count_terms(text, LOCATION_TERMS)
    n_contact = phone_count + email_count
    n_domain = count_terms(text, DOMAIN_TERMS)
    n_name = count_terms(text, NAME_TERMS)

    raw = (
        1.15 * n_id
        + 0.9 * n_loc
        + 1.05 * n_contact
        + 0.75 * n_domain
        + 0.45 * n_name
        - 1.25
    )
    score = sigmoid(raw)
    features = {
        "n_id": n_id,
        "n_loc": n_loc,
        "n_contact": n_contact,
        "n_domain": n_domain,
        "n_name": n_name,
    }
    return score, features
