"""查询构建器

提供流式 API 构建 Elasticsearch 查询。

使用示例:
    ```python
    from app.services.search.query import QueryBuilder

    # 简单查询
    query = QueryBuilder().match("title", "搜索关键词").build()

    # 复杂查询
    query = (QueryBuilder()
        .must_match("title", "关键词")
        .should_match("content", "可选内容")
        .filter_term("status", "published")
        .range_query("created_at", gte="2023-01-01")
        .aggregate("categories", "terms", field="category")
        .build())
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 查询 DSL 构建 ==============


class QueryBuilder:
    """Elasticsearch 查询构建器

    提供流式 API 构建 Elasticsearch 查询 DSL。
    """

    def __init__(self) -> None:
        """初始化查询构建器"""
        self._must: list[dict[str, Any]] = []
        self._should: list[dict[str, Any]] = []
        self._must_not: list[dict[str, Any]] = []
        self._filter: list[dict[str, Any]] = []
        self._aggregations: dict[str, dict[str, Any]] = {}
        self._sort: list[dict[str, Any]] = []
        self._highlight: dict[str, Any] | None = None
        self._source: bool | list[str] | dict[str, Any] | None = None
        self._size: int | None = None
        self._from_: int = 0
        self._track_total_hits: bool | int = True
        self._minimum_should_match: int | str | None = None

    # ============== 布尔查询 ==============

    def must(self, query: dict[str, Any]) -> Self:
        """添加 MUST 子句（必须匹配）

        Args:
            query: 查询 DSL

        Returns:
            self
        """
        self._must.append(query)
        return self

    def should(self, query: dict[str, Any]) -> Self:
        """添加 SHOULD 子句（可选匹配）

        Args:
            query: 查询 DSL

        Returns:
            self
        """
        self._should.append(query)
        return self

    def must_not(self, query: dict[str, Any]) -> Self:
        """添加 MUST_NOT 子句（必须不匹配）

        Args:
            query: 查询 DSL

        Returns:
            self
        """
        self._must_not.append(query)
        return self

    def filter(self, query: dict[str, Any]) -> Self:
        """添加 FILTER 子句（过滤，不计分）

        Args:
            query: 查询 DSL

        Returns:
            self
        """
        self._filter.append(query)
        return self

    # ============== 基础查询 ==============

    def match(
        self,
        field: str,
        value: str | float,
        operator: Literal["and", "or"] = "or",
        minimum_should_match: int | str | None = None,
    ) -> Self:
        """添加 match 查询

        Args:
            field: 字段名
            value: 查询值
            operator: and/or
            minimum_should_match: 最少匹配数

        Returns:
            self
        """
        query: dict[str, Any] = {"match": {field: {"query": value}}}
        if operator != "or":
            query["match"][field]["operator"] = operator
        if minimum_should_match:
            query["match"][field]["minimum_should_match"] = minimum_should_match

        return self.must(query)

    def must_match(
        self,
        field: str,
        value: str | float,
        operator: Literal["and", "or"] = "or",
    ) -> Self:
        """添加 MUST match 查询（便捷方法）

        Args:
            field: 字段名
            value: 查询值
            operator: and/or

        Returns:
            self
        """
        query = {"match": {field: {"query": value, "operator": operator}}}
        return self.must(query)

    def should_match(
        self,
        field: str,
        value: str | float,
        operator: Literal["and", "or"] = "or",
    ) -> Self:
        """添加 SHOULD match 查询（便捷方法）

        Args:
            field: 字段名
            value: 查询值
            operator: and/or

        Returns:
            self
        """
        query = {"match": {field: {"query": value, "operator": operator}}}
        return self.should(query)

    def multi_match(
        self,
        query: str,
        fields: list[str],
        type_: Literal["best_fields", "most_fields", "cross_fields", "phrase", "phrase_prefix"] = "best_fields",
    ) -> Self:
        """添加 multi_match 查询

        Args:
            query: 查询字符串
            fields: 字段列表
            type_: 查询类型

        Returns:
            self
        """
        return self.must({
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": type_,
            }
        })

    def term(self, field: str, value: Any) -> Self:
        """添加 term 查询（精确匹配）

        Args:
            field: 字段名
            value: 查询值

        Returns:
            self
        """
        return self.filter({"term": {field: value}})

    def terms(self, field: str, values: list[Any]) -> Self:
        """添加 terms 查询（多值精确匹配）

        Args:
            field: 字段名
            values: 查询值列表

        Returns:
            self
        """
        return self.filter({"terms": {field: values}})

    def range_query(
        self,
        field: str,
        gt: Any = None,
        gte: Any = None,
        lt: Any = None,
        lte: Any = None,
        format_: str | None = None,
    ) -> Self:
        """添加 range 查询

        Args:
            field: 字段名
            gt: 大于
            gte: 大于等于
            lt: 小于
            lte: 小于等于
            format_: 日期格式

        Returns:
            self
        """
        range_spec: dict[str, Any] = {}
        if gt is not None:
            range_spec["gt"] = gt
        if gte is not None:
            range_spec["gte"] = gte
        if lt is not None:
            range_spec["lt"] = lt
        if lte is not None:
            range_spec["lte"] = lte
        if format_:
            range_spec["format"] = format_

        return self.filter({"range": {field: range_spec}})

    def exists(self, field: str) -> Self:
        """添加 exists 查询（字段存在）

        Args:
            field: 字段名

        Returns:
            self
        """
        return self.filter({"exists": {"field": field}})

    def prefix(self, field: str, value: str) -> Self:
        """添加 prefix 查询（前缀匹配）

        Args:
            field: 字段名
            value: 前缀值

        Returns:
            self
        """
        return self.must({"prefix": {field: value}})

    def wildcard(self, field: str, value: str) -> Self:
        """添加 wildcard 查询（通配符匹配）

        Args:
            field: 字段名
            value: 通配符值（支持 * 和 ?）

        Returns:
            self
        """
        return self.must({"wildcard": {field: value}})

    def fuzzy(
        self,
        field: str,
        value: str,
        fuzziness: str | int = "AUTO",
    ) -> Self:
        """添加 fuzzy 查询（模糊匹配）

        Args:
            field: 字段名
            value: 查询值
            fuzziness: 模糊度

        Returns:
            self
        """
        return self.must({
            "fuzzy": {
                field: {
                    "value": value,
                    "fuzziness": fuzziness,
                }
            }
        })

    # ============== 特殊查询 ==============

    def query_string(
        self,
        query: str,
        default_field: str | None = None,
        default_operator: Literal["and", "or"] = "or",
    ) -> Self:
        """添加 query_string 查询（Lucene 查询语法）

        Args:
            query: 查询字符串
            default_field: 默认字段
            default_operator: 默认操作符

        Returns:
            self
        """
        spec: dict[str, Any] = {"query": query}
        if default_field:
            spec["default_field"] = default_field
        if default_operator != "or":
            spec["default_operator"] = default_operator

        return self.must({"query_string": spec})

    def simple_query_string(
        self,
        query: str,
        fields: list[str] | None = None,
    ) -> Self:
        """添加 simple_query_string 查询（安全版本）

        Args:
            query: 查询字符串
            fields: 字段列表

        Returns:
            self
        """
        spec: dict[str, Any] = {"query": query}
        if fields:
            spec["fields"] = fields

        return self.must({"simple_query_string": spec})

    # ============== 地理查询 ==============

    def geo_distance(
        self,
        field: str,
        lat: float,
        lon: float,
        distance: str,
    ) -> Self:
        """添加 geo_distance 查询

        Args:
            field: 地理位置字段
            lat: 纬度
            lon: 经度
            distance: 距离（如 "10km"）

        Returns:
            self
        """
        return self.filter({
            "geo_distance": {
                "distance": distance,
                field: {"lat": lat, "lon": lon},
            }
        })

    def geo_bounding_box(
        self,
        field: str,
        top: float,
        left: float,
        bottom: float,
        right: float,
    ) -> Self:
        """添加 geo_bounding_box 查询

        Args:
            field: 地理位置字段
            top: 上边界纬度
            left: 左边界经度
            bottom: 下边界纬度
            right: 右边界经度

        Returns:
            self
        """
        return self.filter({
            "geo_bounding_box": {
                field: {
                    "top_left": {"lat": top, "lon": left},
                    "bottom_right": {"lat": bottom, "lon": right},
                }
            }
        })

    # ============== 排序 ==============

    def sort(
        self,
        field: str,
        order: Literal["asc", "desc"] = "desc",
    ) -> Self:
        """添加排序

        Args:
            field: 字段名
            order: 排序方向

        Returns:
            self
        """
        self._sort.append({field: {"order": order}})
        return self

    def sort_score(self, order: Literal["asc", "desc"] = "desc") -> Self:
        """按分数排序

        Args:
            order: 排序方向

        Returns:
            self
        """
        self._sort.append({"_score": {"order": order}})
        return self

    def sort_geo_distance(
        self,
        field: str,
        lat: float,
        lon: float,
        order: Literal["asc", "desc"] = "asc",
    ) -> Self:
        """按地理距离排序

        Args:
            field: 地理位置字段
            lat: 纬度
            lon: 经度
            order: 排序方向

        Returns:
            self
        """
        self._sort.append({
            "_geo_distance": {
                "field": field,
                "location": {"lat": lat, "lon": lon},
                "order": order,
            }
        })
        return self

    # ============== 分页 ==============

    def size(self, size: int) -> Self:
        """设置返回结果数

        Args:
            size: 结果数

        Returns:
            self
        """
        self._size = size
        return self

    def from_(self, from_: int) -> Self:
        """设置偏移量

        Args:
            from_: 偏移量

        Returns:
            self
        """
        self._from_ = from_
        return self

    def paginate(self, page: int, page_size: int) -> Self:
        """设置分页

        Args:
            page: 页码（从 1 开始）
            page_size: 每页大小

        Returns:
            self
        """
        self._from_ = (page - 1) * page_size
        self._size = page_size
        return self

    # ============== 高亮 ==============

    def highlight(
        self,
        fields: list[str] | dict[str, Any],
        pre_tags: list[str] | None = None,
        post_tags: list[str] | None = None,
        fragment_size: int = 150,
        number_of_fragments: int = 3,
    ) -> Self:
        """配置高亮

        Args:
            fields: 字段列表或字段配置
            pre_tags: 前置标签
            post_tags: 后置标签
            fragment_size: 片段大小
            number_of_fragments: 片段数量

        Returns:
            self
        """
        self._highlight = {
            "fields": {},
            "fragment_size": fragment_size,
            "number_of_fragments": number_of_fragments,
        }

        if pre_tags:
            self._highlight["pre_tags"] = pre_tags
        if post_tags:
            self._highlight["post_tags"] = post_tags

        if isinstance(fields, dict):
            self._highlight["fields"] = fields
        else:
            for field in fields:
                self._highlight["fields"][field] = {}

        return self

    # ============== 返回字段 ==============

    def source(
        self,
        includes: list[str] | None = None,
        excludes: list[str] | None = None,
    ) -> Self:
        """配置返回字段

        Args:
            includes: 包含字段
            excludes: 排除字段

        Returns:
            self
        """
        if includes is not None and excludes is not None:
            self._source = {"includes": includes, "excludes": excludes}
        elif includes:
            self._source = includes
        elif excludes:
            self._source = {"excludes": excludes}
        else:
            self._source = False

        return self

    # ============== 聚合 ==============

    def aggregate(
        self,
        name: str,
        type_: Literal[
            "terms",
            "range",
            "date_range",
            "histogram",
            "date_histogram",
            "stats",
            "avg",
            "sum",
            "min",
            "max",
            "cardinality",
            "filters",
            "nested",
        ],
        **kwargs,
    ) -> Self:
        """添加聚合

        Args:
            name: 聚合名称
            type_: 聚合类型
            **kwargs: 聚合参数

        Returns:
            self
        """
        self._aggregations[name] = {type_: kwargs}
        return self

    def terms_aggregation(
        self,
        name: str,
        field: str,
        size: int = 10,
        order: dict[str, Any] | None = None,
    ) -> Self:
        """添加 terms 聚合

        Args:
            name: 聚合名称
            field: 字段名
            size: 返回值数量
            order: 排序

        Returns:
            self
        """
        spec: dict[str, Any] = {"field": field, "size": size}
        if order:
            spec["order"] = order

        return self.aggregate(name, "terms", **spec)

    def date_histogram_aggregation(
        self,
        name: str,
        field: str,
        calendar_interval: Literal["minute", "hour", "day", "week", "month", "quarter", "year"] = "day",
        format_: str | None = None,
    ) -> Self:
        """添加日期直方图聚合

        Args:
            name: 聚合名称
            field: 字段名
            calendar_interval: 时间间隔
            format_: 日期格式

        Returns:
            self
        """
        spec: dict[str, Any] = {"field": field, "calendar_interval": calendar_interval}
        if format_:
            spec["format"] = format_

        return self.aggregate(name, "date_histogram", **spec)

    def range_aggregation(
        self,
        name: str,
        field: str,
        ranges: list[dict[str, Any]],
    ) -> Self:
        """添加范围聚合

        Args:
            name: 聚合名称
            field: 字段名
            ranges: 范围列表 [{"key": "名", "from": 值, "to": 值}]

        Returns:
            self
        """
        return self.aggregate(name, "range", field=field, ranges=ranges)

    def stats_aggregation(self, name: str, field: str) -> Self:
        """添加统计聚合

        Args:
            name: 聚合名称
            field: 字段名

        Returns:
            self
        """
        return self.aggregate(name, "stats", field=field)

    def nested_aggregation(
        self,
        name: str,
        path: str,
        aggs: dict[str, Any] | None = None,
    ) -> Self:
        """添加嵌套聚合

        Args:
            name: 聚合名称
            path: 嵌套字段路径
            aggs: 子聚合

        Returns:
            self
        """
        nested_spec: dict[str, Any] = {"path": path}
        if aggs:
            nested_spec["aggs"] = aggs

        self._aggregations[name] = {"nested": nested_spec}
        return self

    # ============== 其他选项 ==============

    def minimum_should_match(self, value: int | str) -> Self:
        """设置 minimum_should_match

        Args:
            value: 最少匹配数

        Returns:
            self
        """
        self._minimum_should_match = value
        return self

    def track_total_hits(self, value: bool | int) -> Self:
        """设置 track_total_hits

        Args:
            value: 是否精确统计

        Returns:
            self
        """
        self._track_total_hits = value
        return self

    # ============== 构建 ==============

    def build(self) -> dict[str, Any]:
        """构建查询 DSL

        Returns:
            查询字典
        """
        query: dict[str, Any] = {}

        # 构建布尔查询
        if self._must or self._should or self._must_not or self._filter:
            bool_query: dict[str, Any] = {}
            if self._must:
                bool_query["must"] = self._must
            if self._should:
                bool_query["should"] = self._should
            if self._must_not:
                bool_query["must_not"] = self._must_not
            if self._filter:
                bool_query["filter"] = self._filter
            if self._minimum_should_match:
                bool_query["minimum_should_match"] = self._minimum_should_match

            query["bool"] = bool_query
        elif not query:
            # 如果没有任何子句，使用 match_all
            query["match_all"] = {}

        body: dict[str, Any] = {"query": query}

        # 添加其他选项
        if self._aggregations:
            body["aggs"] = self._aggregations
        if self._sort:
            body["sort"] = self._sort
        if self._highlight:
            body["highlight"] = self._highlight
        if self._source is not None:
            body["_source"] = self._source
        if self._size is not None:
            body["size"] = self._size
        if self._from_ > 0:
            body["from"] = self._from_
        if self._track_total_hits is not True:
            body["track_total_hits"] = self._track_total_hits

        return body

    def build_query_only(self) -> dict[str, Any]:
        """仅构建查询部分（不含分页、排序等）

        Returns:
            查询字典
        """
        body = self.build()
        return body.get("query", {"match_all": {}})


# ============== 搜索结果类 ==============


@dataclass
class SearchResult:
    """搜索结果

    Attributes:
        id: 文档 ID
        score: 相关度分数
        source: 文档内容
        highlight: 高亮结果
        index: 索引名称
    """
    id: str
    score: float
    source: dict[str, Any]
    highlight: dict[str, list[str]] | None = None
    index: str | None = None


@dataclass
class SearchResponse:
    """搜索响应

    Attributes:
        hits: 搜索结果列表
        total: 总结果数
        max_score: 最高分数
        aggregations: 聚合结果
        took: 耗时（毫秒）
    """
    hits: list[SearchResult]
    total: int
    max_score: float | None = None
    aggregations: dict[str, Any] | None = None
    took: int = 0

    @classmethod
    def from_es_response(cls, response: dict[str, Any]) -> SearchResponse:
        """从 Elasticsearch 响应构建

        Args:
            response: Elasticsearch 原始响应

        Returns:
            SearchResponse 实例
        """
        hits_info = response.get("hits", {})
        hits_raw = hits_info.get("hits", [])

        hits = []
        for hit in hits_raw:
            hits.append(SearchResult(
                id=hit.get("_id", ""),
                score=hit.get("_score", 0.0),
                source=hit.get("_source", {}),
                highlight=hit.get("highlight"),
                index=hit.get("_index"),
            ))

        total_info = hits_info.get("total", {})
        total = total_info.get("value", len(hits))

        return cls(
            hits=hits,
            total=total,
            max_score=hits_info.get("max_score"),
            aggregations=response.get("aggregations"),
            took=response.get("took", 0),
        )


@dataclass
class AggregationBucket:
    """聚合桶

    Attributes:
        key: 桶键
        key_as_string: 键的字符串表示
        doc_count: 文档数量
        sub_aggregations: 子聚合
    """
    key: str | int | float | None
    key_as_string: str | None = None
    doc_count: int = 0
    sub_aggregations: dict[str, Any] = field(default_factory=dict)


def parse_terms_aggregation(response: dict[str, Any], name: str) -> list[AggregationBucket]:
    """解析 terms 聚合结果

    Args:
        response: 聚合响应
        name: 聚合名称

    Returns:
        桶列表
    """
    agg = response.get(name, {})
    buckets = agg.get("buckets", [])

    result = []
    for bucket in buckets:
        result.append(AggregationBucket(
            key=bucket.get("key"),
            key_as_string=bucket.get("key_as_string"),
            doc_count=bucket.get("doc_count", 0),
            sub_aggregations={k: v for k, v in bucket.items() if k not in ("key", "key_as_string", "doc_count")},
        ))

    return result


def parse_date_histogram_aggregation(response: dict[str, Any], name: str) -> list[AggregationBucket]:
    """解析日期直方图聚合结果

    Args:
        response: 聚合响应
        name: 聚合名称

    Returns:
        桶列表
    """
    return parse_terms_aggregation(response, name)
