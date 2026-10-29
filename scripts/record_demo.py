"""录制 3 分钟心语 Demo（无旁白，屏幕内嵌字幕卡）。

前置：
  - API 运行在 http://localhost:8010
  - Web 运行在 http://localhost:3000
  - playwright 已安装（pip install playwright；首次需 playwright install chromium）

用法：
  python scripts/record_demo.py
  python scripts/record_demo.py --base http://localhost:8080

产物：
  assets/demo/heart-say-demo.mp4
  assets/demo/heart-say-demo.srt
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "demo"
RAW_DIR = OUT_DIR / ".raw"

OVERLAY_JS = """
(args) => {
  const old = document.getElementById('demo-caption');
  if (old) old.remove();
  const styleId = 'demo-caption-style';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      #demo-caption {
        position: fixed; right: 20px; bottom: 20px; width: 274px;
        padding: 16px 17px 15px; z-index: 2147483000;
        border: 1px solid rgba(159,188,174,.22); border-radius: 16px;
        background: rgba(9,14,19,.92); backdrop-filter: blur(20px);
        box-shadow: 0 18px 50px rgba(0,0,0,.35);
        animation: demoIn .45s cubic-bezier(.22,1,.36,1) both;
        font-family: "PingFang SC","Hiragino Sans GB",sans-serif;
      }
      #demo-caption .demo-eyebrow {
        margin: 0 0 8px; font-family: "SFMono-Regular",monospace;
        font-size: 8px; letter-spacing: .18em; color: #9fbcae;
      }
      #demo-caption .demo-title {
        margin: 0 0 7px; font-family: "Songti SC","Noto Serif SC",serif;
        font-size: 20px; font-weight: 300; letter-spacing: .06em; color: #e7edee;
      }
      #demo-caption .demo-sub {
        margin: 0; font-size: 10.5px; line-height: 1.75; color: #8a99a3;
      }
      #demo-caption .demo-progress {
        margin-top: 11px; height: 2px; background: rgba(255,255,255,.06); border-radius: 99px; overflow: hidden;
      }
      #demo-caption .demo-progress span {
        display: block; height: 100%; width: var(--demo-progress, 10%);
        background: linear-gradient(90deg, rgba(159,188,174,.3), #9fbcae);
      }
      @keyframes demoIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    `;
    document.head.appendChild(style);
  }
  const box = document.createElement('div');
  box.id = 'demo-caption';
  box.innerHTML = `
    <p class="demo-eyebrow">${args.eyebrow}</p>
    <p class="demo-title">${args.title}</p>
    <p class="demo-sub">${args.sub}</p>
    <div class="demo-progress"><span style="--demo-progress:${args.progress}%"></span></div>
  `;
  document.body.appendChild(box);
}
"""


def caption(page, index: int, total: int, eyebrow: str, title: str, sub: str) -> None:
    page.evaluate(
        OVERLAY_JS,
        {
            "eyebrow": eyebrow,
            "title": title,
            "sub": sub,
            "progress": round(100 * index / total),
        },
    )


def wait_service(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"服务返回 {response.status}")
    except Exception as exc:
        raise SystemExit(f"服务不可用：{url}（{exc}）。请先启动 API 与 Web。") from exc


def fmt_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:3000", help="Web 地址")
    parser.add_argument("--api", default="http://localhost:8010", help="API 健康检查地址")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    wait_service(args.api.rstrip("/") + "/health")
    wait_service(args.base)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    captions: list[dict] = []
    scene = {"index": 0}
    started = time.monotonic()

    def show(eyebrow: str, title: str, sub: str) -> None:
        if captions:
            captions[-1]["end"] = time.monotonic() - started
        scene["index"] += 1
        captions.append(
            {
                "start": time.monotonic() - started,
                "end": None,
                "eyebrow": eyebrow,
                "title": title,
                "sub": sub,
            }
        )
        caption(page, scene["index"], 9, eyebrow, title, sub)

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=1,
            record_video_dir=str(RAW_DIR),
            record_video_size={"width": args.width, "height": args.height},
        )
        page = context.new_page()
        page.goto(args.base, wait_until="networkidle")
        page.wait_for_selector(".hero", timeout=20000)

        show(
            "DEMO 01 / HEART SAY",
            "先让情绪有地方落下。",
            "面向中国家庭的青少年心理教育与情绪陪伴。不诊断、不荐药，先共情，再给一个能走的小步。",
        )
        page.wait_for_timeout(24000)

        show(
            "02 / LISTEN",
            "先共情，再给一个小步骤。",
            "家长说孩子不想上学。心语先接住焦虑，再把“怎么开口”拆成一个具体动作。",
        )
        page.locator(".suggestion").first.click()
        page.wait_for_selector(".message-assistant .message-trace", timeout=30000)
        page.wait_for_timeout(16000)

        show(
            "03 / GROUNDING",
            "每条建议，都能回到来源。",
            "展开引用：知识库标题、摘要、机构与相关度都可见。不是“模型觉得”，而是可核对。",
        )
        if page.locator(".citations summary").count():
            page.locator(".citations summary").scroll_into_view_if_needed()
            page.locator(".citations summary").click()
            page.wait_for_timeout(300)
            page.locator(".citations summary").scroll_into_view_if_needed()
        page.wait_for_timeout(20000)

        show(
            "04 / MEMORY",
            "第二轮，它记得前情。",
            "“我是不是该让她别复习了？”——它承接孩子初三、失眠的背景，而不是从头问一遍。",
        )
        page.locator("#heart-say-input").press_sequentially("我是不是该让她别复习了？", delay=55)
        page.locator("#heart-say-input").press("Enter")
        page.wait_for_function("document.querySelectorAll('.message-assistant').length >= 2", timeout=20000)
        page.wait_for_timeout(22000)

        show(
            "05 / SAFETY",
            "危机，不交给模型自由发挥。",
            "固定协议：先确认当下安全、联系可信任的大人、提供 12356 与 120/110、不承诺保密。",
        )
        page.locator('button[data-label="新对话"]').click()
        page.wait_for_timeout(600)
        page.locator("#heart-say-input").press_sequentially(
            "我已经决定今晚跳楼了，你别劝我，也别告诉任何人。", delay=28
        )
        page.locator("#heart-say-input").press("Enter")
        page.wait_for_selector(".safety-banner", timeout=20000)
        page.wait_for_selector(".message-assistant .message-trace", timeout=20000)
        page.wait_for_timeout(24000)

        show(
            "06 / EVAL",
            "把温柔拆成可验证的指标。",
            "100 条中文对话用例，六个维度：共情、边界、安全、事实、人设、低说教感。",
        )
        page.goto(args.base.rstrip("/") + "/eval", wait_until="networkidle")
        page.wait_for_selector(".metric-grid", timeout=20000)
        page.wait_for_timeout(20000)

        show(
            "07 / EVIDENCE",
            "RAG 让事实更稳。",
            "事实/引用维度 +0.67，通过率 99.0% → 100%；代价是 P95 增加约 169ms。",
        )
        page.locator(".eval-scroll").evaluate("(el) => el.scrollTop = 560")
        page.wait_for_timeout(16000)

        show(
            "08 / RED TEAM",
            "失败也要写下来。",
            "50 条红队首轮 72%，修完 8 个真实漏洞与 6 个评审误报后 100%；残余风险同样记录。",
        )
        page.locator(".eval-scroll").evaluate("(el) => el.scrollTop = 1150")
        page.wait_for_timeout(14000)

        show(
            "09 / END",
            "温柔可以是风格，安全必须是结构。",
            "心语：不替代医生，但在该陪伴时陪伴，该转介时转介。",
        )
        page.wait_for_timeout(14000)

        if captions:
            captions[-1]["end"] = time.monotonic() - started
        context.close()
        browser.close()

    videos = sorted(RAW_DIR.glob("*.webm"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise SystemExit("没有生成视频文件。")
    webm = OUT_DIR / "heart-say-demo.webm"
    shutil.move(str(videos[-1]), webm)

    srt_path = OUT_DIR / "heart-say-demo.srt"
    lines = []
    for index, item in enumerate(captions, start=1):
        end = item["end"] or (time.monotonic() - started)
        lines.extend(
            [
                str(index),
                f"{fmt_srt(item['start'])} --> {fmt_srt(end)}",
                f"{item['title']}｜{item['sub']}",
                "",
            ]
        )
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    mp4 = OUT_DIR / "heart-say-demo.mp4"
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        os.system(
            f'"{ffmpeg}" -y -i "{webm}" -c:v libx264 -preset medium -crf 21 '
            f'-pix_fmt yuv420p -movflags +faststart -an "{mp4}" >/dev/null 2>&1'
        )
        if mp4.exists() and mp4.stat().st_size > 0:
            webm.unlink(missing_ok=True)
            print(f"MP4: {mp4}")
        else:
            print(f"WebM（未找到 ffmpeg，保留原文件）: {webm}")
    except Exception as exc:
        print(f"WebM（转换失败，保留原文件）：{webm}（{exc}）")

    shutil.rmtree(RAW_DIR, ignore_errors=True)
    print(f"SRT: {srt_path}")
    print(f"时长约 {captions[-1]['end']:.0f} 秒，字幕 {len(captions)} 条。")


if __name__ == "__main__":
    main()
