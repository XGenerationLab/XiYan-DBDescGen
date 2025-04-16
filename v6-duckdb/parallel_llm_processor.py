# parallel_llm_processor.py
"""并行LLM处理模块

该模块支持并行调用大语言模型API，提高批量处理效率。
它处理请求限速、错误重试，并提供结果缓存。
"""

import time
import hashlib
import json
import os
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.llms import LLM
from call_llamaindex_llm import call_llm, call_llm_with_json_validation


class ParallelLLMProcessor:
    """并行LLM处理器，支持批量并行处理LLM请求"""

    def __init__(self, llm: LLM, max_workers: int = 5,
                 rate_limit: float = 0.5, retry_delay: float = 2.0,
                 max_retries: int = 3, cache_dir: Optional[str] = "./llm_cache",
                 use_cache: bool = True):
        """初始化并行LLM处理器

        Args:
            llm: 大语言模型实例
            max_workers: 最大并行工作线程数
            rate_limit: 请求间隔时间(秒)
            retry_delay: 重试延迟时间(秒)
            max_retries: 最大重试次数
            cache_dir: 缓存目录路径
            use_cache: 是否使用缓存
        """
        self.llm = llm
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # 创建缓存目录
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # 用于请求限速的时间戳
        self.last_request_time = 0

    def _calculate_cache_key(self, prompt: str, **kwargs) -> str:
        """计算缓存键

        Args:
            prompt: LLM提示文本
            **kwargs: 其他参数

        Returns:
            缓存键
        """
        # 创建一个包含所有相关内容的字符串
        cache_str = f"{prompt}_{json.dumps(kwargs, sort_keys=True)}"

        # 使用MD5生成一个固定长度的哈希
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存中获取结果

        Args:
            cache_key: 缓存键

        Returns:
            缓存的结果，如果不存在则返回None
        """
        if not self.use_cache:
            return None

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"读取缓存时出错: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """保存结果到缓存

        Args:
            cache_key: 缓存键
            result: 要缓存的结果
        """
        if not self.use_cache:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(result)
        except Exception as e:
            print(f"保存缓存时出错: {e}")

    def _rate_limited_call(self, func: Callable, *args, **kwargs) -> Any:
        """使用速率限制调用函数

        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数调用结果
        """
        # 检查距离上次请求的时间
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # 如果需要，等待以满足速率限制
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        # 更新最后请求时间
        self.last_request_time = time.time()

        # 调用函数
        return func(*args, **kwargs)

    def _process_single_prompt(self, prompt: str, task_id: Optional[str] = None,
                               with_json_validation: bool = False,
                               required_fields: Optional[List[str]] = None,
                               expected_type: Optional[type] = None,
                               **kwargs) -> Union[str, Dict[str, Any]]:
        """处理单个提示

        Args:
            prompt: LLM提示文本
            task_id: 任务ID(可选)
            with_json_validation: 是否进行JSON验证
            required_fields: JSON必需字段列表
            expected_type: 预期的JSON类型
            **kwargs: 其他LLM调用参数

        Returns:
            LLM响应或JSON验证结果
        """
        # 计算缓存键
        cache_key = self._calculate_cache_key(prompt, **kwargs)

        # 尝试从缓存获取
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            print(f"任务 {task_id or '未命名'}: 从缓存获取结果")
            if with_json_validation:
                # 尝试解析缓存的JSON结果
                try:
                    return json.loads(cached_result)
                except:
                    # 缓存的结果不是有效JSON，继续处理
                    pass
            else:
                return cached_result

        # 执行LLM调用，带重试
        for retry in range(self.max_retries):
            try:
                if with_json_validation:
                    result = self._rate_limited_call(
                        call_llm_with_json_validation,
                        prompt=prompt,
                        llm=self.llm,
                        required_fields=required_fields,
                        expected_type=expected_type,
                        max_retries=1,  # 我们在这里自己处理重试
                        **kwargs
                    )

                    # 对于JSON验证，我们只缓存成功的结果
                    if result.get("success", False):
                        self._save_to_cache(cache_key, json.dumps(result.get("data", {})))

                    return result.get("data") if result.get("success", False) else {}

                else:
                    result = self._rate_limited_call(call_llm, prompt, self.llm, **kwargs)
                    # 缓存结果
                    self._save_to_cache(cache_key, result)
                    return result

            except Exception as e:
                print(f"任务 {task_id or '未命名'} 第 {retry + 1}/{self.max_retries} 次重试出错: {e}")
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay * (retry + 1))  # 指数退避

        # 所有重试都失败
        print(f"任务 {task_id or '未命名'} 在 {self.max_retries} 次尝试后失败")
        return "" if not with_json_validation else {}

    def process_batch(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并行处理一批提示

        Args:
            prompts: 提示配置列表，每项应包含:
                     - prompt: 提示文本
                     - task_id: 任务ID(可选)
                     - with_json_validation: 是否需要JSON验证(可选)
                     - required_fields: JSON验证所需字段(可选)
                     - expected_type: 预期JSON类型(可选)
                     - kwargs: 其他参数(可选)

        Returns:
            结果列表，每项包含:
                     - task_id: 任务ID
                     - result: LLM响应
                     - success: 是否成功
        """
        results = []

        # 使用ThreadPoolExecutor并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_prompt = {}
            for i, prompt_config in enumerate(prompts):
                task_id = prompt_config.get("task_id", f"task_{i}")

                # 提取参数
                prompt_text = prompt_config.get("prompt", "")
                with_json_validation = prompt_config.get("with_json_validation", False)
                required_fields = prompt_config.get("required_fields")
                expected_type = prompt_config.get("expected_type")
                kwargs = prompt_config.get("kwargs", {})

                # 提交任务
                future = executor.submit(
                    self._process_single_prompt,
                    prompt=prompt_text,
                    task_id=task_id,
                    with_json_validation=with_json_validation,
                    required_fields=required_fields,
                    expected_type=expected_type,
                    **kwargs
                )

                future_to_prompt[future] = {
                    "task_id": task_id,
                    "config": prompt_config
                }

            # 收集结果
            for future in as_completed(future_to_prompt):
                prompt_info = future_to_prompt[future]
                task_id = prompt_info["task_id"]

                try:
                    result = future.result()
                    results.append({
                        "task_id": task_id,
                        "result": result,
                        "success": True
                    })
                    print(f"任务 {task_id} 成功完成")
                except Exception as e:
                    results.append({
                        "task_id": task_id,
                        "result": None,
                        "success": False,
                        "error": str(e)
                    })
                    print(f"任务 {task_id} 处理失败: {e}")

        # 按原始顺序排序结果
        task_id_to_index = {prompt.get("task_id", f"task_{i}"): i for i, prompt in enumerate(prompts)}
        results.sort(key=lambda x: task_id_to_index.get(x["task_id"], 0))

        return results


# 使用示例:
"""
# 初始化处理器
processor = ParallelLLMProcessor(llm=your_llm_instance, max_workers=5)

# 准备一批提示
prompts = [
    {
        "task_id": "table_analysis_1",
        "prompt": "分析表结构...",
        "with_json_validation": True,
        "expected_type": dict,
        "required_fields": ["table_type", "fields"]
    },
    {
        "task_id": "table_analysis_2",
        "prompt": "分析另一个表结构...",
        "with_json_validation": True,
        "expected_type": dict,
        "required_fields": ["table_type", "fields"]
    },
    # 更多提示...
]

# 并行处理
results = processor.process_batch(prompts)

# 使用结果
for result in results:
    if result["success"]:
        # 处理成功的结果
        print(f"任务 {result['task_id']} 结果: {result['result']}")
    else:
        # 处理失败
        print(f"任务 {result['task_id']} 失败: {result.get('error')}")
"""