"""纸张计数推理：直接修改下方配置，命令行只需要可选的 --all。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter


# ==================== 运行前修改这里 ====================
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT.parent / "数据" / "5.png"  # 单张图片，或递归处理的图片目录
OUTPUT_PATH = ROOT / "output" / "扑克牌检测"  # 结果放在其 compact/ 或 all/ 下
CONFIG_PATH = ROOT / "configs" / "regions.yaml"  # tracks；框检测改为 regions.yaml
ROTATE_CCW = False  # 横向拍摄的 tracks 图片通常需改为 True（逆时针旋转90度）
# ======================================================

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true",
        help="输出全部中间过程图和诊断数据；默认只输出检测图、坐标及基本数据",
    )
    parser.add_argument('--recipe',help='Versioned recipe name or YAML path; replaces CONFIG_PATH')
    args = parser.parse_args(argv)

    input_path = Path(INPUT_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()
    target = output_path / ("all" if args.all else "compact")
    try:
        if input_path == output_path or output_path in input_path.parents:
            raise ValueError("输入路径不能位于输出目录内，请修改 INPUT_PATH 或 OUTPUT_PATH")
        if input_path.is_file():
            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"不支持的图片格式：{input_path.suffix}")
            images = [input_path]
        elif input_path.is_dir():
            images = sorted(
                p for p in input_path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                and output_path not in p.parents
            )
            if not images:
                raise ValueError(f"输入目录中没有支持的图片：{input_path}")
        else:
            raise ValueError(f"输入路径不存在，请修改脚本开头的 INPUT_PATH：{input_path}")

        # 放在参数解析后导入，使 --help 不需要加载算法和图像依赖。
        import matplotlib
        matplotlib.use("Agg")
        from paper_counter import PaperDetector
        from paper_counter.compact_output import save_compact
        from paper_counter.visualization import save_debug

        detector = PaperDetector(recipe=args.recipe) if args.recipe else PaperDetector(config=CONFIG_PATH)
        exporter = save_debug if args.all or detector.recipe.runtime.export=='full_debug' else save_compact
        target.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    rows = []
    print(f"模式：{'全部过程' if args.all else '精简结果'}；图片数：{len(images)}")
    for index, image_path in enumerate(images, 1):
        # 保留相对目录及扩展名，避免 a.bmp 与 a.png 的结果互相覆盖。
        destination = target if input_path.is_file() else target / image_path.relative_to(input_path)
        start = perf_counter()
        row = {"image": str(image_path), "output": str(destination)}
        try:
            result = detector.detect(image_path, rotate_ccw=ROTATE_CCW)
            exporter(result.raw_result, destination, detector.config)
            row.update(
                success=True, count=result.count, confidence=result.confidence,
                status=result.status, warnings=result.warnings, period=result.period,
                algorithm=detector.config.algorithm,
                rotation_ccw_degrees=result.rotation_ccw_degrees,
            )
            print(f"[{index}/{len(images)}] {image_path.name}：{result.count}，状态：{result.status}")
            for warning in result.warnings:
                print(f"  提示：{warning}")
        except Exception as error:
            row.update(success=False, error=str(error))
            print(f"[{index}/{len(images)}] {image_path.name} 失败：{error}", file=sys.stderr)
        row["elapsed_seconds"] = round(perf_counter() - start, 3)
        rows.append(row)

    summary = {
        "export_mode": "all" if args.all else "compact",
        "confidence_interpretation": "未校准的一致性评分，不是正确概率",
        "total": len(rows), "succeeded": sum(row["success"] for row in rows),
        "failed": sum(not row["success"] for row in rows), "results": rows,
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(f"完成：成功 {summary['succeeded']}，失败 {summary['failed']}。输出：{target}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
