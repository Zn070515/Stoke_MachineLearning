"""CSRC 证监会 industry-gate (门类) mapping for the PIT sector membership.

The 证监会 industry classification (门类, single-letter A–S) is stable across
the 2001 / 2012 / 中国上市公司协会 renames of the standard, so a stock's gate
letter can be merged across all three CNINFO ``分类标准`` labels
(§v19 P0#1).  This map is pure (stdlib only) so the data layer and the
downloader share it without pulling torch.
"""
from __future__ import annotations

#: CSRC 门类 name (as CNINFO's ``行业门类`` reports it) → gate letter.
CSRC_GATE_CODES: dict[str, str] = {
    "农、林、牧、渔业": "A",
    "采矿业": "B",
    "制造业": "C",
    "电力、热力、燃气及水生产和供应业": "D",
    "建筑业": "E",
    "批发和零售业": "F",
    "交通运输、仓储和邮政业": "G",
    "住宿和餐饮业": "H",
    "信息传输、软件和信息技术服务业": "I",
    "金融业": "J",
    "房地产业": "K",
    "租赁和商务服务业": "L",
    "科学研究和技术服务业": "M",
    "水利、环境和公共设施管理业": "N",
    "居民服务、修理和其他服务业": "O",
    "教育": "P",
    "卫生和社会工作": "Q",
    "文化、体育和娱乐业": "R",
    "综合": "S",
    "制造业门类": "C",   # 2001-standard spelling variant
    "金融业门类": "J",
}

#: CNINFO ``分类标准`` labels that all denote the CSRC standard family.
CSRC_STANDARD_LABELS: frozenset[str] = frozenset({
    "证监会行业分类标准（2001）",
    "证监会行业分类标准（2012）",
    "中国上市公司协会上市公司行业分类标准",
})


def csrc_gate_code(gate_name: str) -> str | None:
    """Gate letter for a ``行业门类`` name, or None when unrecognized."""
    return CSRC_GATE_CODES.get(gate_name)
