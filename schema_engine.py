from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, text
from sqlalchemy.engine import Engine
from llama_index.core import SQLDatabase
from llama_index.core.llms import LLM
from components import (
    field_category,
    generate_column_desc,
    generate_table_desc,
    understand_fields_by_category,
    understand_database,
    understand_date_time_min_gran,
    dummy_sql_generator
)
from utils import examples_to_str
from type_engine import TypeEngine
from mschema import MSchema


class SchemaEngine(SQLDatabase):
    def __init__(self, engine: Engine, schema: Optional[str] = None, metadata: Optional[MetaData] = None,
                 ignore_tables: Optional[List[str]] = None, include_tables: Optional[List[str]] = None,
                 sample_rows_in_table_info: int = 3, indexes_in_table_info: bool = False,
                 custom_table_info: Optional[dict] = None, view_support: bool = False, max_string_length: int = 300,
                 mschema: Optional[MSchema] = None, llm: Optional[LLM] = None,
                 db_name: Optional[str] = '', comment_mode: str = 'origin'):
        super().__init__(engine, schema, metadata, ignore_tables, include_tables, sample_rows_in_table_info,
                         indexes_in_table_info, custom_table_info, view_support, max_string_length)

        self._db_name = db_name
        self._usable_tables = [table_name for table_name in self._usable_tables if self._inspector.has_table(table_name, schema)]
        self._dialect = engine.dialect.name
        self._type_engine = TypeEngine()
        assert self._dialect in self._type_engine.supported_dialects, "Unsupported dialect {}.".format(self._dialect)

        self._llm = llm

        if mschema is not None:
            self._mschema = mschema
        else:
            self._mschema = MSchema(db_id=db_name, schema=schema, type_engine=self._type_engine)
            self.init_mschema()

        self.comment_mode = comment_mode

    @property
    def mschema(self) -> MSchema:
        """Return M-Schema"""
        return self._mschema

    @property
    def type_engine(self) -> TypeEngine:
        return self._type_engine

    def get_pk_constraint(self, table_name: str) -> Dict:
        return self._inspector.get_pk_constraint(table_name, self._schema)['constrained_columns']

    def get_table_comment(self, table_name: str):
        try:
            return self._inspector.get_table_comment(table_name, self._schema)['text']
        except: # sqlite不支持添加注释
            return ''

    def default_schema_name(self) -> Optional[str]:
        return self._inspector.default_schema_name

    def get_schema_names(self) -> List[str]:
        return self._inspector.get_schema_names()

    def get_table_options(self, table_name: str) -> Dict[str, Any]:
        return self._inspector.get_table_options(table_name, self._schema)

    def get_foreign_keys(self, table_name: str):
        return self._inspector.get_foreign_keys(table_name, self._schema)

    def get_unique_constraints(self, table_name: str):
        # 唯一键
        return self._inspector.get_unique_constraints(table_name, self._schema)

    def get_indexes(self, table_name: str):
        # 索引字段
        return self._inspector.get_indexes(table_name, self._schema)

    def add_semicolon_to_sql(self, sql_query: str):
        if not sql_query.strip().endswith(';'):
            sql_query += ';'
        return sql_query

    def fetch(self, sql_query: str):
        sql_query = self.add_semicolon_to_sql(sql_query)

        with self._engine.begin() as connection:
            try:
                cursor = connection.execute(text(sql_query))
                records = cursor.fetchall()
            except Exception as e:
                print("An exception occurred during SQL execution.\n", e)
                records = None
            return records

    def fetch_truncated(self, sql_query: str, max_rows: Optional[int] = None, max_str_len: int = 30) -> Dict:
        sql_query = self.add_semicolon_to_sql(sql_query)
        with self._engine.begin() as connection:
            try:
                cursor = connection.execute(text(sql_query))
                result = cursor.fetchall()
                truncated_results = []
                if max_rows:
                    result = result[:max_rows]
                for row in result:
                    truncated_row = tuple(
                        self.truncate_word(column, length=max_str_len)
                        for column in row
                    )
                    truncated_results.append(truncated_row)
                return {"truncated_results": truncated_results, "fields": list(cursor.keys())}
            except Exception as e:
                print("An exception occurred during SQL execution.\n", e)
                records = None
                return {"truncated_results": records, "fields": []}

    def trunc_result_to_markdown(self, sql_res: Dict) -> str:
        """
        数据库查询结果转换成markdown格式
        """
        truncated_results = sql_res.get("truncated_results", [])
        fields = sql_res.get("fields", [])

        if not truncated_results:
            return ""

        header = "| " + " | ".join(fields) + " |"
        separator = "| " + " | ".join(["---"] * len(fields)) + " |"
        rows = []
        for row in truncated_results:
            rows.append("| " + " | ".join(str(value) for value in row) + " |")
        markdown_table = "\n".join([header, separator] + rows)
        return markdown_table

    def execute(self, sql_query: str, timeout=10) -> Any:
        sql_query = self.add_semicolon_to_sql(sql_query)
        def run_query():
            with self._engine.begin() as connection:
                cursor = connection.execute(text(sql_query))
                return True
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_query)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                print(f"SQL执行超时（{timeout}秒）{sql_query}.")
                return None
            except Exception as e:
                print("执行SQL时发生异常。", e)
                return None

    def get_protected_table_name(self, table_name: str) -> str:
        if self._dialect == self._type_engine.mysql_dialect or self._dialect == self._type_engine.sqlite_dialect:
            return f'`{table_name}`'
        elif self._dialect == self._type_engine.postgres_dialect:
            if self._schema:
                return f'"{self._schema}"."{table_name}"'
            else:
                return f'"{table_name}"'
        elif self._dialect == self._type_engine.sqlserver_dialect:
            return f'[{table_name}]'
        else:
            raise NotImplementedError

    def get_protected_field_name(self, field_name: str) -> str:
        if self._dialect == self._type_engine.mysql_dialect or self._dialect == self._type_engine.sqlite_dialect:
            return f'`{field_name}`'
        elif self._dialect == self._type_engine.postgres_dialect:
            return f'"{field_name}"'
        else:
            raise NotImplementedError

    def init_mschema(self):
        for table_name in self._usable_tables:
            table_comment = self.get_table_comment(table_name)
            table_comment = '' if table_comment is None else table_comment.strip()
            self._mschema.add_table(table_name, fields={}, comment=table_comment)
            pks = self.get_pk_constraint(table_name)

            # 数据表的唯一键
            unique_keys = []
            unique_constraints = self.get_unique_constraints(table_name)
            for u_con in unique_constraints:
                column_names = u_con['column_names']
                unique_keys.append(column_names)
            self._mschema.tables[table_name]['unique_keys'] = unique_keys

            # 数据表索引
            indexes = self.get_indexes(table_name)
            keys = []
            for index in indexes:
                is_unique = index.get("unique", False)
                keys.append(index['column_names'])
            self._mschema.tables[table_name]['keys'] = keys

            fks = self.get_foreign_keys(table_name)
            constrained_columns = []
            for fk in fks:
                referred_schema = fk['referred_schema']
                for c, r in zip(fk['constrained_columns'], fk['referred_columns']):
                    self._mschema.add_foreign_key(table_name, c, referred_schema, fk['referred_table'], r)
                    constrained_columns.append(c)

            fields = self._inspector.get_columns(table_name, schema=self._schema)
            for field in fields:
                field_type = f"{field['type']!s}"
                field_name = field['name']
                if field_name in pks:
                    primary_key = True
                    if len(pks) == 1:
                        is_unique = True
                    else:
                        is_unique = False
                else:
                    primary_key = False
                    if [field_name] in unique_keys:
                        is_unique = True
                    else:
                        is_unique = False
                field_comment = field.get("comment", None)
                field_comment = "" if field_comment is None else field_comment.strip()
                autoincrement = field.get('autoincrement', False)
                default = field.get('default', None)
                if default is not None:
                    default = f'{default}'

                examples = []
                try:
                    sql = f"select distinct {self.get_protected_field_name(field_name)} from {self.get_protected_table_name(table_name)} where {self.get_protected_field_name(field_name)} is not null limit 5;"
                    examples = [s[0] for s in self.fetch(sql)]
                except:
                    pass
                examples = examples_to_str(examples)
                if None in examples:
                    examples.remove(None)
                if '' in examples:
                    examples.remove('')

                self._mschema.add_field(table_name, field_name, field_type=field_type, primary_key=primary_key,
                    nullable=field['nullable'], default=default, autoincrement=autoincrement, unique=is_unique,
                    comment=field_comment, examples=examples)

    def get_column_count(self, table_name: str, field_name: str) -> int:
        sql = 'select count({}) from {};'.format(self.get_protected_field_name(field_name),
                                                 self.get_protected_table_name(table_name))
        r = self.fetch(sql)
        if r is not None:
            total_num = r[0][0]
        else:
            total_num = -1
        return total_num

    def get_column_unique_count(self, table_name: str, field_name: str) -> int:
        sql = 'select count(distinct {}) from {};'.format(self.get_protected_field_name(field_name),
                                                          self.get_protected_table_name(table_name))
        r = self.fetch(sql)
        if r is not None:
            unique_num = r[0][0]
        else:
            unique_num = -1
        return unique_num

    def get_column_value_examples(self, table_name: str, field_name: str, max_rows: Optional[int] = None, max_str_len: int = 30)-> List:
        sql = 'select distinct {} from {} where {} is not null;'.format(self.get_protected_field_name(field_name),
            self.get_protected_table_name(table_name), self.get_protected_field_name(field_name))
        res = self.fetch_truncated(sql, max_rows, max_str_len)
        res = res['truncated_results']
        if res is not None:
            return [r[0] for r in res]
        else:
            return []

    def check_column_value_exist(self, table_name: str, field_name: str, value_name: str, is_string: bool) -> bool:
        if is_string:
            sql = '''select count(*) from {} where {} = '{}';'''.format(
                self.get_protected_table_name(table_name), self.get_protected_field_name(field_name), value_name.replace("'", "''"))
        else:
            sql = "select count(*) from {} where {} = {};".format(
                self.get_protected_table_name(table_name), self.get_protected_field_name(field_name), value_name)
        r = self.fetch(sql)
        if r is not None:
            return r[0][0] > 0
        else:
            return False

    def check_agg_func(self, agg_func: str):
        assert agg_func.upper() in ['MAX', 'MIN', 'AVG', 'SUM'], \
            "Invalid aggregate function {}.".format(agg_func)

    def get_column_agg_value(self, table_name: str, field_name: str, field_type: str, agg_func: str):
        self.check_agg_func(agg_func)
        if self._type_engine.field_type_cate(field_type) != self._type_engine.field_type_number_label:
            return None

        sql = 'select {}({}) from {} where {} is not null;'.format(agg_func, self.get_protected_field_name(field_name),
            self.get_protected_table_name(table_name), self.get_protected_field_name(field_name))
        r = self.fetch(sql)
        if r is not None:
            return r[0][0]
        else:
            return None

    def get_column_agg_char_length(self, table_name: str, field_name: str, agg_func: str) -> int:
        if self._dialect == self._type_engine.postgres_dialect:
            snip = '{}::TEXT'.format(self.get_protected_field_name(field_name))
        elif self._dialect == self._type_engine.mysql_dialect:
            snip = '{}'.format(self.get_protected_field_name(field_name))
        elif self._dialect == self._type_engine.sqlite_dialect:
            snip = "CAST({} AS TEXT)".format(self.get_protected_field_name(field_name))
        elif self._dialect == self._type_engine.sqlserver_dialect:
            snip = 'CAST({} AS NVARCHAR(MAX))'.format(self.get_protected_field_name(field_name))
        else:
            raise NotImplementedError

        self.check_agg_func(agg_func)
        if self._dialect == self._type_engine.sqlite_dialect:
            sql = 'select {}(length({})) from {} where {} is not null;'.format(agg_func, snip,
                self.get_protected_table_name(table_name),self.get_protected_field_name(field_name))
        elif self._dialect == self._type_engine.sqlserver_dialect:
            sql = 'select {}(LEN({})) from {} where {} is not null;'.format(agg_func, snip,
                self.get_protected_table_name(table_name), self.get_protected_field_name(field_name))
        else:
            sql = 'select {}(char_length({})) from {} where {} is not null;'.format(agg_func, snip,
                self.get_protected_table_name(table_name), self.get_protected_field_name(field_name))
        r = self.fetch(sql)
        if r is not None and r[0][0] is not None:
            return r[0][0]
        else:
            return -1

    def get_all_field_examples(self, table_name: str,  max_rows: Optional[int] = None):
        sql = f"""SELECT DISTINCT * FROM {self.get_protected_table_name(table_name)}"""  # group by {dimension_fields}
        if max_rows is not None and max_rows > 0:
            sql += ' LIMIT {};'.format(max_rows)
        return sql

    def get_single_field_info_str(self, table_name: str, field_name: str)->str:
        """
        某一列的相关信息：列名、类型、列描述、是否主键、最大/最小值等
        """
        field_info = self._mschema.get_field_info(table_name, field_name)
        field_type = field_info.get('type', '')

        unique_num = self.get_column_unique_count(table_name, field_name)
        total_num = self.get_column_count(table_name, field_name)
        max_value = self.get_column_agg_value(table_name, field_name, field_type, 'max')
        min_value = self.get_column_agg_value(table_name, field_name, field_type, 'min')
        avg_value = self.get_column_agg_value(table_name, field_name, field_type, 'avg')
        max_len = self.get_column_agg_char_length(table_name, field_name, 'max')
        min_len = self.get_column_agg_char_length(table_name, field_name, 'min')

        comment = field_info.get('comment', '')
        primary_key = field_info.get("primary_key", False)
        nullable = field_info.get("nullable", True)

        field_info_str = ['【字段信息】', f'字段名称: {field_name}', f'字段类型: {field_type}']
        dim_or_meas = field_info.get('dim_or_meas', '')
        unique = field_info.get('unique', False)
        if primary_key:
            unique = True

        if len(comment) > 0:
            field_info_str.append(f'字段描述: {comment}')
        field_info_str.append(f'是否为主键(或者与其他字段组成联合主键): {primary_key}')
        field_info_str.append(f'UNIQUE: {unique}')
        field_info_str.append(f'NULLABLE: {nullable}')
        date_min_gran = field_info.get('date_min_gran', None)
        if total_num >= 0:
            field_info_str.append(f'COUNT: {total_num}')
        if unique_num >= 0:
            field_info_str.append(f'COUNT(DISTINCT): {unique_num}')
        if max_value is not None:
            field_info_str.append(f'MAX: {max_value}')
        if min_value is not None:
            field_info_str.append(f'MIN: {min_value}')
        if avg_value is not None:
            field_info_str.append(f'AVG: {avg_value}')
        if max_len >= 0:
            field_info_str.append(f'MAX(CHAR_LENGTH): {max_len}')
        if min_len >= 0:
            field_info_str.append(f'MIN(CHAR_LENGTH): {min_len}')
        if dim_or_meas in self._type_engine.dim_measure_labels:
            field_info_str.append(f'Dimension/Measure: {dim_or_meas}')
        if date_min_gran is not None:
            field_info_str.append(f'该字段表示的语义可能与日期或时间有关，推测它表示的最小时间颗粒度是: {date_min_gran}')

        value_examples = self.get_column_value_examples(table_name, field_name, max_rows=10, max_str_len=30)
        if len(value_examples) > 0:
            field_info_str.append(f"Value Examples: {value_examples}")

        return '\n'.join(field_info_str)

    def fields_category(self):
        tables = self._mschema.tables
        for table_name in tables.keys():
            print("Table Name: ", table_name)
            fields = tables[table_name]['fields']
            for field_name, field_info in fields.items():
                print("Field Name: ", field_name)
                field_type = field_info['type']
                field_type_cate = self._type_engine.field_type_cate(field_type)
                field_info_str = self.get_single_field_info_str(table_name, field_name)
                res = field_category(field_type_cate, self._type_engine, self._llm, field_info_str=field_info_str)
                print(field_info_str)
                print(res)
                if res['category'] == self._type_engine.field_category_date_label:
                    min_gran = understand_date_time_min_gran(field_info_str, llm=self._llm)
                    print("最小时间颗粒度：", min_gran)
                    if min_gran in self._type_engine.date_time_min_grans:
                        self._mschema.set_column_property(table_name, field_name, "date_min_gran", min_gran)

                category = res['category']
                # 对于枚举类型的字段，获取它所有的枚举候选值
                if category == self._type_engine.field_category_enum_label:
                    examples = self.get_column_value_examples(table_name, field_name)
                    examples = [s for s in examples if len(str(examples)) > 0]
                    self._mschema.set_column_property(table_name, field_name, "examples", examples)
                self._mschema.set_column_property(table_name, field_name, "category", res['category'])
                self._mschema.set_column_property(table_name, field_name, "dim_or_meas", res['dim_or_meas'])


    def table_and_column_desc_generation(self, language: str='CN'):
        """"
        table and column description genration

        四种模式：
        no_comment: 不带任何描述信息
        origin: 跟数据库中保持一致
        generation: 清除已有的描述信息，完全由模型生成
        merge: 没有描述的生成描述信息；已经有描述信息的，不再生成
        """
        if self.comment_mode == 'origin':
            return
        elif self.comment_mode == 'merge':
            pass
        elif self.comment_mode == 'generation':
            self._mschema.erase_all_column_comment()
            self._mschema.erase_all_table_comment()
        elif self.comment_mode == 'no_comment':
            self._mschema.erase_all_column_comment()
            self._mschema.erase_all_table_comment()
            return
        else:
            raise NotImplementedError(f"Unsupported comment mode {self.comment_mode}.")

        db_mschema = self._mschema.to_mschema()
        """1、初步理解数据库的基本信息和每张表的内容"""
        db_info = understand_database(db_mschema, self._llm)
        self._mschema.db_info = db_info
        print("DB INFO: ", db_info)

        for table_name, table_info in self._mschema.tables.items():
            fields = table_info['fields']
            table_comment = table_info.get('comment', '')
            if len(table_comment) >= 10:
                need_table_comment = False
            else:
                need_table_comment = True

            table_mschema = self._mschema.single_table_mschema(table_name)

            sql = self.get_all_field_examples(table_name, max_rows=10)
            res = self.fetch_truncated(sql, max_rows=10)
            res = self.trunc_result_to_markdown(res)

            """2、按照维度和度量分类，理解各个维度/度量字段之间的区别与联系，供参考"""
            supp_info = {}
            dim_fields = self._mschema.get_dim_or_meas_fields(self._type_engine.dimension_label, table_name)
            mea_fields = self._mschema.get_dim_or_meas_fields(self._type_engine.measure_label, table_name)
            if len(dim_fields) > 0:
                supp_info[self._type_engine.dimension_label] = understand_fields_by_category(db_info, table_name,
                    table_mschema, self._llm, sql, res, dim_fields, self._type_engine.dimension_label)
            if len(mea_fields) > 0:
                supp_info[self._type_engine.measure_label] = understand_fields_by_category(db_info, table_name,
                    table_mschema, self._llm, sql, res, mea_fields, self._type_engine.measure_label)
            print("Supplementary information：")
            print(supp_info)

            """3、对每一列生成列描述"""
            for field_name, field_info in fields.items():
                field_info_str = self.get_single_field_info_str(table_name, field_name)
                dim_or_meas = field_info.get("dim_or_meas", '')
                field_desc = field_info.get('comment', '')
                if len(field_desc) == 0:  # 原来没有字段描述，重新生成
                    field_desc = generate_column_desc(field_name, field_info_str, table_mschema,
                                                      self._llm, sql, res, supp_info.get(dim_or_meas, ""),
                                                      language=language)
                    print("Table Name: {}, Field Name: {}".format(table_name, field_name))
                    print("Column Description: {}".format(field_desc))
                    self._mschema.set_column_property(table_name, field_name, 'comment', field_desc)

            """4、表描述生成"""
            table_mschema = self._mschema.single_table_mschema(table_name)
            if need_table_comment:
                table_desc = generate_table_desc(table_name, table_mschema, self._llm, sql, res, language=language)
                print("Table Description: {}".format(table_desc))
                self._mschema.set_table_property(table_name, 'comment', table_desc)

    def sql_generator(self, question: str, evidence: str = '') -> str:
        db_mschema = self._mschema.to_mschema()
        pred_sql = dummy_sql_generator(self._dialect, db_mschema=db_mschema,
            question=question, evidence=evidence, llm=self._llm)

        return pred_sql
