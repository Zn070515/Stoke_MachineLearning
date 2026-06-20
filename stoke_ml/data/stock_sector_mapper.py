"""Map stock codes to industry sectors for ETF flow data merging.

Uses AKShare to fetch industry classifications, with a local cache
to avoid repeated API calls.
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Simplified industry→sector mapping based on common Chinese
# industry classification names
INDUSTRY_TO_SECTOR = {
    "证券": "券商", "券商": "券商", "证券业": "券商",
    "半导体": "半导体", "集成电路": "半导体", "芯片": "半导体",
    "医药": "医药", "医药制造": "医药", "医疗器械": "医药",
    "生物制药": "医药", "中药": "医药", "化学制药": "医药",
    "白酒": "白酒", "酿酒": "白酒", "酒": "白酒", "饮料": "白酒",
    "食品": "消费", "消费": "消费", "零售": "消费",
    "家用电器": "消费", "纺织": "消费", "服装": "消费",
    "新能源": "新能源", "光伏": "新能源", "风电": "新能源",
    "储能": "新能源", "锂电池": "新能源",
    "银行": "银行", "银行业": "银行",
    "军工": "军工", "国防": "军工", "航天": "军工", "航空": "军工",
    "科技": "科技", "计算机": "科技", "软件": "科技",
    "通信": "科技", "电子": "科技", "5G": "科技",
    "房地产开发": "房地产", "房地产": "房地产", "地产": "房地产",
    "汽车": "汽车", "汽车零部件": "汽车", "整车": "汽车",
    "有色": "有色", "有色金属": "有色", "稀土": "有色",
    "黄金": "有色", "铜": "有色", "铝": "有色",
    "煤炭": "煤炭", "采掘": "煤炭", "能源": "煤炭",
    "钢铁": "钢铁", "钢材": "钢铁",
    "传媒": "传媒", "游戏": "传媒", "影视": "传媒",
    "互联网": "传媒", "广告": "传媒",
    "电力": "电力", "电力设备": "电力", "电网": "电力",
    "化工": "化工", "化学": "化工", "石化": "化工",
    "建筑": "建筑", "基建": "建筑", "工程": "建筑",
    "建材": "建材", "水泥": "建材", "玻璃": "建材",
    "交通": "交通", "运输": "交通", "物流": "交通",
    "农林牧渔": "农业", "农业": "农业", "种业": "农业",
    "保险": "保险", "保险业": "保险",
    "环保": "环保", "碳中和": "环保", "碳交易": "环保",
}


class StockSectorMapper:
    """Map A-share stock codes to industry sectors.

    Uses AKShare for industry data with local CSV caching.
    """

    def __init__(
        self,
        mapping_path: str | None = None,
        cache_path: str | None = None,
    ):
        self._mapping: dict[str, str] = {}
        self._mapping_path = mapping_path
        self._cache_path = cache_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "a_shares", "stock_sector_cache.csv",
        )
        self._loaded = False

    def get_sector(self, stock_code: str) -> str | None:
        """Return the sector name for a stock, or None if unknown."""
        if not self._loaded:
            self._load_cache()
        return self._mapping.get(str(stock_code).zfill(6))

    def _load_cache(self) -> None:
        """Load stock→sector mapping from cache or build from AKShare."""
        # Prefer user-provided mapping_path, then cache, then API
        load_path = self._mapping_path or self._cache_path
        if os.path.exists(load_path):
            try:
                df = pd.read_csv(load_path, dtype=str)
                self._mapping = dict(zip(df["stock_code"], df["sector"]))
                self._loaded = True
                logger.debug("Loaded %d stock→sector mappings from %s",
                             len(self._mapping), load_path)
                return
            except Exception:
                if self._mapping_path:
                    logger.warning("Failed to load mapping_path %s, falling back",
                                   self._mapping_path)

        if self._mapping_path:
            # mapping_path was provided but failed — try cache before API
            if os.path.exists(self._cache_path):
                try:
                    df = pd.read_csv(self._cache_path, dtype=str)
                    self._mapping = dict(zip(df["stock_code"], df["sector"]))
                    self._loaded = True
                    return
                except Exception:
                    pass

        self._build_from_akshare()

    def _build_from_akshare(self) -> None:
        """Fetch industry classifications from AKShare and build mapping."""
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for sector mapping")
            self._loaded = True
            return

        try:
            # Get all industry boards
            industries = ak.stock_board_industry_name_em()
        except Exception as e:
            logger.warning("Failed to fetch industry names: %s", e)
            self._loaded = True
            return

        for _, row in industries.iterrows():
            industry_name = row.get("板块名称", "")
            sector = self._classify_industry(industry_name)
            if sector is None:
                continue

            try:
                stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
            except Exception:
                continue

            for _, srow in stocks.iterrows():
                code = str(srow.get("代码", "")).zfill(6)
                if code and len(code) == 6:
                    self._mapping[code] = sector

        self._loaded = True
        self._save_cache()
        logger.info("Built %d stock→sector mappings from AKShare", len(self._mapping))

    @staticmethod
    def _classify_industry(industry_name: str) -> str | None:
        """Map an industry name to our sector taxonomy."""
        name = str(industry_name)
        for keyword, sector in INDUSTRY_TO_SECTOR.items():
            if keyword in name:
                return sector
        return None

    def _save_cache(self) -> None:
        """Save mapping to local CSV."""
        if not self._mapping:
            return
        df = pd.DataFrame(
            [(k, v) for k, v in self._mapping.items()],
            columns=["stock_code", "sector"],
        )
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        df.to_csv(self._cache_path, index=False)
