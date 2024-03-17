# Error Check Log

Posted on 2024-03-16 by Tungwong Chi

## 1. Error Logs

**Error** :
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

**Check** :
```diff
- dataset_map_fn=oasst1_map_fn,
+ dataset_map_fn=None,
```

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
