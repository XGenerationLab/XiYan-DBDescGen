class TypeEngine:
    def __int__(self):
        pass

    @property
    def supported_dialects(self):
        return [self.mysql_dialect, self.postgres_dialect, self.sqlite_dialect, self.sqlserver_dialect]

    @property
    def mysql_dialect(self):
        return 'mysql'

    @property
    def postgres_dialect(self):
        return 'postgresql'

    @property
    def sqlite_dialect(self):
        return 'sqlite'
        
    @property
    def sqlserver_dialect(self):
        return 'mssql'

    def field_type_abbr(self, field_type: str):
        """字段类型缩写，用于在MSchema中展示"""
        return field_type.split("(")[0]

    @property
    def mysql_date_types(self):
        return ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP', 'YEAR']

    @property
    def pg_date_types(self):
        return ['DATE', 'TIME', 'TIMESTAMP', 'TIMESTAMP WITHOUT TIME ZONE', 'TIMESTAMP WITH TIME ZONE']

    @property
    def date_date_type(self):
        return 'DATE'

    @property
    def date_time_type(self):
        return 'TIME'

    @property
    def date_datetime_type(self):
        return 'DATETIME'

    @property
    def date_timestamp_type(self):
        return 'TIMESTAMP'

    @property
    def all_date_types(self):
        return ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP', 'TIMESTAMP WITHOUT TIME ZONE',
                'TIMESTAMP WITH TIME ZONE']

    @property
    def mysql_string_types(self):
        return ['BLOB', 'TINYBLOB', 'MEDIUMBLOB', 'LONGBLOB', 'CHAR', 'VARCHAR',
                'TEXT', 'TINYTEXT', 'MEDIUMTEXT', 'LONGTEXT']

    @property
    def pg_string_types(self):
        return ['CHARACTER VARYING', 'VARCHAR', 'CHAR', 'CHARACTER', 'TEXT']

    @property
    def all_string_types(self):
        return self.mysql_string_types + self.pg_string_types

    @property
    def mysql_number_types(self):
        return ['TINYINT', 'SMALLINT', 'MEDIUMINT', 'INT', 'INTEGER', 'BIGINT',
                'FLOAT', 'DOUBLE', 'DECIMAL']

    @property
    def pg_number_types(self):
        return ['SMALLINT', 'INTEGER', 'BIGINT',
                'DECIMAL', 'NUMERIC', 'REAL', 'DOUBLE PRECISION',
                'SMALLSERIAL', 'SERIAL', 'BIGSERIAL']

    @property
    def all_number_types(self):
        return self.mysql_number_types + self.pg_number_types

    @property
    def all_enum_types(self):
        return ['ENUM', 'SET']

    def field_type_cate(self, field_type: str) -> str:
        """根据数据类型分组"""
        field_type = self.field_type_abbr(field_type.upper())
        if field_type in self.all_number_types:
            return self.field_type_number_label
        elif field_type in self.all_string_types:
            return self.field_type_string_label
        elif field_type in self.all_date_types:
            return self.field_type_date_label
        elif field_type in ['BOOL', 'BOOLEAN']:
            return self.field_type_bool_label
        else:
            return self.field_type_other_label

    @property
    def date_time_min_grans(self):
        """时间日期类字段的最小颗粒度"""
        return ['YEAR', 'MONTH', 'DAY', 'QUARTER', 'WEEK', 'HOUR', 'MINUTE',
                'SECOND', 'MILLISECOND', 'MICROSECOND', 'OTHER']

    @property
    def field_type_all_labels(self):
        return [self.field_type_number_label, self.field_type_string_label, self.field_type_date_label,
                self.field_type_bool_label, self.field_type_other_label]

    @property
    def field_type_number_label(self):
        return 'Number'

    @property
    def field_type_string_label(self):
        return 'String'

    @property
    def field_type_date_label(self):
        return 'DateTime'

    @property
    def field_type_bool_label(self):
        return 'Bool'

    @property
    def field_type_other_label(self):
        return 'Other'

    @property
    def field_category_all_labels(self):
        return [self.field_category_code_label, self.field_category_enum_label,
                self.field_category_date_label, self.field_category_text_label,
                self.field_category_measure_label]

    @property
    def field_category_code_label(self):
        return 'Code'

    @property
    def field_category_enum_label(self):
        return 'Enum'

    @property
    def field_category_date_label(self):
        return 'DateTime'

    @property
    def field_category_text_label(self):
        return 'Text'

    @property
    def field_category_measure_label(self):
        return 'Measure'

    @property
    def dim_measure_labels(self):
        return [self.dimension_label, self.measure_label]

    @property
    def dimension_label(self):
        return 'Dimension'

    @property
    def measure_label(self):
        return 'Measure'
