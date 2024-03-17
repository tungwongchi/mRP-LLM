# Error Check Log

Posted on 2024-03-16 by Tungwong Chi

## 1. Error Logs

### Error 1

**Description**: KeyError: 'text' in dataset_map_fn

**Error Message**:
```bash
Traceback (most recent call last):
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/multiprocess/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/utils/py_utils.py", line 623, in _write_generator_to_queue
    for i, result in enumerate(func(**kwargs)):
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3458, in _map_single
    example = apply_function_on_filtered_inputs(example, i, offset=offset)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3361, in apply_function_on_filtered_inputs
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
  File "/root/code/mRP-LLM/xtuner/xtuner/dataset/map_fns/dataset_map_fns/oasst1_map_fn.py", line 22, in oasst1_map_fn
    for sentence in example['text'].strip().split('###'):
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/formatting/formatting.py", line 270, in __getitem__
    value = self.data[key]
KeyError: 'text'
```

**Suggested Fix**:
```diff
- dataset_map_fn=oasst1_map_fn,
+ dataset_map_fn=None,
```

### Error 2

**Description**: requires the protobuf library

**Error Message**:
```bash
03/17 10:48:44 - mmengine - WARNING - Failed to search registry with scope "mmengine" in the "builder" registry tree. As a workaround, the current "builder" registry in "xtuner" is used to build instance. This may cause unexpected failure when running the built modules. Please check whether "mmengine" is a correct scope, or whether the registry is initialized.
Traceback (most recent call last):
  File "/root/code/mRP-LLM/xtuner/xtuner/tools/train.py", line 307, in <module>
    main()
  File "/root/code/mRP-LLM/xtuner/xtuner/tools/train.py", line 300, in main
    runner = RUNNERS.build(cfg)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 196, in build_runner_from_cfg
    runner = runner_cls.from_cfg(args)  # type: ignore
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 423, in from_cfg
    runner = cls(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 403, in __init__
    self.register_hooks(default_hooks, custom_hooks)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 1430, in register_hooks
    self.register_custom_hooks(custom_hooks)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 1410, in register_custom_hooks
    self.register_hook(hook)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 1310, in register_hook
    hook_obj = HOOKS.build(hook)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg
    obj = obj_cls(**args)  # type: ignore
  File "/root/code/mRP-LLM/xtuner/xtuner/engine/hooks/dataset_info_hook.py", line 24, in __init__
    self.tokenizer = BUILDER.build(tokenizer)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg
    obj = obj_cls(**args)  # type: ignore
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 801, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2029, in from_pretrained
    return cls._from_pretrained(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2261, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/internlm2-chat-1_8b/tokenization_internlm2_fast.py", line 131, in __init__
    super().__init__(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/tokenization_utils_fast.py", line 114, in __init__
    fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/convert_slow_tokenizer.py", line 1386, in convert_slow_tokenizer
    return converter_class(transformer_tokenizer).converted()
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/convert_slow_tokenizer.py", line 501, in __init__
    requires_backends(self, "protobuf")
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1292, in requires_backends
    raise ImportError("".join(failed))
ImportError: 
InternLM2Converter requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
```

**Suggested Fix**:
```bash
pip install protobuf
```

### Error 3

**Description**: FileNotFoundError: Unable to find '/root/code/mRP-LLM/tatsu-lab/alpaca'

**Error Message**:
```bash
Traceback (most recent call last):
  File "/root/code/mRP-LLM/xtuner/xtuner/tools/train.py", line 307, in <module>
    main()
  File "/root/code/mRP-LLM/xtuner/xtuner/tools/train.py", line 303, in main
    runner.train()
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 1160, in train
    self._train_loop = self.build_train_loop(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 958, in build_train_loop
    loop = LOOPS.build(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg
    obj = obj_cls(**args)  # type: ignore
  File "/root/code/mRP-LLM/xtuner/xtuner/engine/runner/loops.py", line 32, in __init__
    dataloader = runner.build_dataloader(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 824, in build_dataloader
    dataset = DATASETS.build(dataset_cfg)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg
    obj = obj_cls(**args)  # type: ignore
  File "/root/code/mRP-LLM/xtuner/xtuner/dataset/huggingface.py", line 299, in process_hf_dataset
    return process(**kwargs)
  File "/root/code/mRP-LLM/xtuner/xtuner/dataset/huggingface.py", line 167, in process
    dataset = build_origin_dataset(dataset, split)
  File "/root/code/mRP-LLM/xtuner/xtuner/dataset/huggingface.py", line 30, in build_origin_dataset
    dataset = BUILDER.build(dataset)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg
    obj = obj_cls(**args)  # type: ignore
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/load.py", line 2523, in load_dataset
    builder_instance = load_dataset_builder(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/load.py", line 2195, in load_dataset_builder
    dataset_module = dataset_module_factory(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/load.py", line 1736, in dataset_module_factory
    ).get_module()
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/load.py", line 1120, in get_module
    data_files = DataFilesDict.from_patterns(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/data_files.py", line 689, in from_patterns
    DataFilesList.from_patterns(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/data_files.py", line 594, in from_patterns
    resolve_pattern(
  File "/root/code/mRP-LLM/conda_env/lib/python3.10/site-packages/datasets/data_files.py", line 383, in resolve_pattern
    raise FileNotFoundError(error_msg)
FileNotFoundError: Unable to find '/root/code/mRP-LLM/tatsu-lab/alpaca'
```

**Suggested Fix**:
```diff
- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = './data/xiyou.jsonl'
```

### Error 4

**Description**: st.session_state has no attribute "messages"

**Error Message**:
```bash
Traceback (most recent call last):
  File "/root/code/mRP-LLM/web_demo.py", line 290, in <module>
    main()
  File "/root/code/mRP-LLM/web_demo.py", line 251, in main
    for message in st.session_state.messages:
  File "/root/.local/lib/python3.10/site-packages/streamlit/runtime/state/session_state_proxy.py", line 121, in __getattr__
    raise AttributeError(_missing_attr_error_message(key))
AttributeError: st.session_state has no attribute "messages". Did you forget to initialize it? More info: https://docs.streamlit.io/library/advanced-features/session-state#initialization
```

**Suggested Fix**:
```bash
streamlit run web_demo.py 
```


<!-- 
### Error 1

**Description**: [Add a brief description of the next error]

**Error Message**:
```bash
[Insert the error message here]
```

**Suggested Fix**:
```diff
[Insert the suggested fix here]
```
-->

## 2. References

...

## Appendix

### How to Cite This Article

To reference this article, please use the following formats:

```bibtex
@online{mRP-LLM,
    title={Multi role-playing large language model ECL},
    author={Tungwong Chi},
    year={2024},
    month={03},
    url={\url{https://github.com/tungwongchi/mRP-LLM}},
}
```

---

&copy; 2020 Tungwong Chi. All rights reserved.
