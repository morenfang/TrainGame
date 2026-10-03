"""让 pytest 从项目根目录运行时可 import core / render / app。

放在根目录的 conftest.py 会被 pytest 自动加载，这里显式把项目根插入 sys.path，
避免依赖 pytest 的 import 模式配置。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
