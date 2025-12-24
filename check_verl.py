
# try:
#     import flash_attn
#     print('Flash attention installed:', flash_attn.__version__)
# except ImportError:
#     print('Flash attention not installed')

import transformers
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
print('Available classes:')
import inspect
classes = [name for name, obj in inspect.getmembers(modeling_qwen2_5_vl) if inspect.isclass(obj)]
for cls in classes:
    if 'Qwen' in cls or 'Attention' in cls:
        print(f'  {cls}')