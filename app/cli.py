"""Командная строка.

  python -m app.cli models                      — проверить связь с Open WebUI, список моделей
  python -m app.cli check фото.jpg [папка ...]  — проверить файлы, JSON в stdout
  python -m app.cli testset --dir testset       — прогнать тестовый набор и сравнить с эталоном
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .config import load_settings
from .llm import LLMError
from .prompt import FAILED, SUCCESS, TEST_TYPES
from .service import InvalidImageError, PhotoChecker

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _collect(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"не найдено: {p}", file=sys.stderr)
    return files


async def _run_many(checker: PhotoChecker, files: List[Path], concurrency: int) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(f: Path) -> Dict[str, Any]:
        async with sem:
            try:
                res = await checker.check(f.read_bytes(), f.name)
                mark = "✔" if res["overall"] == SUCCESS else "✘"
                print(f"  {mark} {f.name}  ({res['meta']['latency_ms']} мс)", file=sys.stderr)
                return res
            except (InvalidImageError, LLMError) as e:
                print(f"  ! {f.name}: {e}", file=sys.stderr)
                return {"error": str(e), "meta": {"file": f.name}}

    return await asyncio.gather(*(one(f) for f in files))


async def cmd_models(args) -> int:
    checker = PhotoChecker(load_settings())
    models = await checker.llm.list_models()
    for m in models:
        mark = "*" if m.get("id") == checker.s.model else " "
        print(f"{mark} {m.get('id')}  {m.get('name', '')}")
    if checker.s.model not in [m.get("id") for m in models]:
        print(f"\nВнимание: модель LLM_MODEL={checker.s.model} не найдена", file=sys.stderr)
        return 1
    return 0


async def cmd_check(args) -> int:
    checker = PhotoChecker(load_settings())
    files = _collect(args.paths)
    if not files:
        print("нет изображений для проверки", file=sys.stderr)
        return 2
    results = await _run_many(checker, files, args.concurrency)
    if not args.meta:
        for r in results:
            r.pop("meta", None) if "error" not in r else None
    out = results[0] if len(results) == 1 else results
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"результат: {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0 if all(r.get("overall") == SUCCESS for r in results) else 1


async def cmd_testset(args) -> int:
    root = Path(args.dir)
    cases = json.loads((root / "expected.json").read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if args.only in c["id"] or args.only == c["requirement"]]
    checker = PhotoChecker(load_settings())
    print(f"Тестовый набор: {len(cases)} фото, модель {checker.s.model}\n", file=sys.stderr)
    results = await _run_many(checker, [root / c["file"] for c in cases], args.concurrency)

    rows, per_type = [], defaultdict(lambda: [0, 0])  # test_type -> [совпало, всего]
    verdict_ok = 0
    for case, res in zip(cases, results):
        row: Dict[str, Any] = {"id": case["id"], "requirement": case["requirement"], "label": case["label"]}
        if "error" in res:
            row.update(passed=False, error=res["error"])
            rows.append(row)
            continue
        got = {c["test_type"]: c for c in res["checks"]}
        mismatches = []
        for tt, exp in case.get("expect", {}).items():
            per_type[tt][1] += 1
            if got[tt]["result"] == exp:
                per_type[tt][0] += 1
            else:
                mismatches.append(f"{tt}: ожидалось {exp}, получено {got[tt]['result']}")
        any_failed = case.get("expect_failed_any")
        if any_failed and not any(got[t]["result"] == FAILED for t in any_failed):
            mismatches.append("ожидался failed хотя бы в одном из: " + ", ".join(any_failed))
        exp_overall = case.get("expect_overall")
        if exp_overall:
            if res["overall"] == exp_overall:
                verdict_ok += 1
            else:
                mismatches.append(f"итог: ожидалось {exp_overall}, получено {res['overall']}")
        row.update(
            passed=not mismatches,
            mismatches=mismatches,
            overall=res["overall"],
            failed=[f"{c['test_type']}: {c['comment']}" for c in res["checks"] if c["result"] == FAILED],
            latency_ms=res["meta"]["latency_ms"],
        )
        rows.append(row)

    passed = sum(r["passed"] for r in rows)
    with_overall = sum(1 for c in cases if c.get("expect_overall"))
    summary = {
        "model": checker.s.model,
        "cases": len(rows),
        "cases_passed": passed,
        "overall_verdict_accuracy": f"{verdict_ok}/{with_overall}",
        "per_test_type": {t: f"{per_type[t][0]}/{per_type[t][1]}" for t in TEST_TYPES if per_type[t][1]},
    }

    print("\n" + "=" * 78, file=sys.stderr)
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}", file=sys.stderr)
        for m in r.get("mismatches", []) + ([r["error"]] if "error" in r else []):
            print(f"        - {m}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f"Совпало с эталоном: {passed}/{len(rows)} фото; итоговый вердикт: {summary['overall_verdict_accuracy']}",
          file=sys.stderr)
    for t, v in summary["per_test_type"].items():
        print(f"  {t:<16} {v}", file=sys.stderr)

    report = {"summary": summary, "cases": rows, "raw": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out}", file=sys.stderr)
    return 0 if passed == len(rows) else 1


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Проверка фото на пропуск через LLM")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="список моделей Open WebUI")

    c = sub.add_parser("check", help="проверить файлы или папки")
    c.add_argument("paths", nargs="+")
    c.add_argument("--out", help="сохранить JSON в файл")
    c.add_argument("--no-meta", dest="meta", action="store_false", help="без служебного блока meta")
    c.add_argument("--concurrency", type=int, default=2)

    t = sub.add_parser("testset", help="прогнать тестовый набор")
    t.add_argument("--dir", default="testset")
    t.add_argument("--out", default="results/report.json")
    t.add_argument("--only", help="фильтр: часть id или test_type требования")
    t.add_argument("--concurrency", type=int, default=2)

    args = p.parse_args(argv)
    handler = {"models": cmd_models, "check": cmd_check, "testset": cmd_testset}[args.cmd]
    try:
        return asyncio.run(handler(args))
    except LLMError as e:
        print(f"Ошибка LLM: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
