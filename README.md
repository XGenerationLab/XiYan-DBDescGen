# Automatic database description generation for Text-to-SQL

## Important Links

🤖[Arxiv](https://arxiv.org/abs/2502.20657) |
📖[XiYan-SQL](https://github.com/XGenerationLab/XiYan-SQL) |


## Introduction
This repository provides a method for automatically generating effective database descriptions when explicit descriptions are unavailable. The proposed method employs a dual-process approach: a coarse-to-fine process, followed by a fine-to-coarse process. Experimental results on the Bird benchmark indicate that using descriptions generated by the proposed improves SQL generation accuracy by 0.93% compared to not using descriptions, and achieves 37% of human-level performance. 
We support three common database dialects: SQLite, MySQL, PostgreSQL and SQL Server.

Read more: [Arxiv](https://arxiv.org/abs/2502.20657)
<p align="center">
  <img src="https://github.com/XGenerationLab/XiYan-DBDescGen/blob/main/description_generation.png" alt="image" width="1000"/>
</p>

## Requirements
+ python >= 3.9

You can install the required packages with the following command:
```shell
pip install -r requirements.txt
```

## Quick Start

1. Create a database connection.

Connect to SQLite:
```python
import os
from sqlalchemy import create_engine

db_path = "path_to_sqlite"
abs_path = os.path.abspath(db_path)
db_engine = create_engine(f'sqlite:///{abs_path}')
```

2. Set llama-index LLM.

Take dashscope as an example:
```python
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_PLUS, api_key='YOUR API KEY HERE.')
```

3. Generate the database description and build M-Schema.
```python
from schema_engine import SchemaEngine

db_name = 'your_db_name'
comment_mode = 'generation'
schema_engine_instance = SchemaEngine(db_engine, llm=dashscope_llm, db_name=db_name,
                                      comment_mode=comment_mode)
schema_engine_instance.fields_category()
schema_engine_instance.table_and_column_desc_generation()
mschema = schema_engine_instance.mschema
mschema.save(f'./{db_name}.json')
mschema_str = mschema.to_mschema()
print(mschema_str)
```

## Citation
If you find our work helpful, feel free to give us a cite.

```bibtex
@article{description_generation,
      title={Automatic database description generation for Text-to-SQL}, 
      author={Yingqi Gao and Zhiling Luo},
      year={2025},
      eprint={2502.20657},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.20657}, 
}

@article{XiYanSQL,
      title={XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL}, 
      author={Yifu Liu and Yin Zhu and Yingqi Gao and Zhiling Luo and Xiaoxia Li and Xiaorong Shi and Yuntao Hong and Jinyang Gao and Yu Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2507.04701},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04701}, 
}

@article{xiyansql_pre,
      title={A Preview of XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL}, 
      author={Yingqi Gao and Yifu Liu and Xiaoxia Li and Xiaorong Shi and Yin Zhu and Yiming Wang and Shiqi Li and Wei Li and Yuntao Hong and Zhiling Luo and Jinyang Gao and Liyu Mou and Yu Li},
      year={2024},
      journal={arXiv preprint arXiv:2411.08599},
      url={https://arxiv.org/abs/2411.08599},
      primaryClass={cs.AI}
}
```
