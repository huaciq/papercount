"""Compatibility entry for the regions batch; all batch logic lives in batch_detect."""
from pathlib import Path
import sys
from batch_detect import main

if __name__=='__main__':
    arguments=sys.argv[1:]
    if '--config' not in arguments:
        arguments += ['--config',str(Path(__file__).resolve().parents[1]/'configs/regions.yaml')]
    if '--datasets' not in arguments and '--input' not in arguments:
        arguments += ['--datasets','测试集','扑克牌']
    raise SystemExit(main(arguments))
