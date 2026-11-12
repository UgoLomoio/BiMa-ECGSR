"""Screenshots of every tab of a running app (headless Chrome via Playwright).

  python space/tools/screenshots.py --url http://127.0.0.1:7862/ --out <dir> [--chrome /usr/bin/google-chrome]
"""
import argparse, os, time

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:7862/"); ap.add_argument("--out", required=True)
ap.add_argument("--chrome", default="/usr/bin/google-chrome"); ap.add_argument("--example", default="AFIB")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
log = []


def wait_text(page, text, timeout=180):
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout * 1000)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path=a.chrome, headless=True, args=["--no-sandbox", "--disable-gpu", "--no-proxy-server"])
    page = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1)
    page.goto(a.url, wait_until="networkidle", timeout=120000)
    wait_text(page, "Research demonstration")
    time.sleep(2)
    page.screenshot(path=os.path.join(a.out, "tab0_landing.png"), full_page=True)
    # choose an example whose label starts with --example
    try:
        page.get_by_label("Example record").click(); time.sleep(0.5)
        page.get_by_role("option").filter(has_text=a.example + " -").first.click(); time.sleep(0.5)
        log.append(f"example {a.example} selected")
    except Exception as exc:
        log.append(f"example selection failed ({type(exc).__name__}); default example used")
        page.keyboard.press("Escape")
    # Tab 1
    t0 = time.time(); page.get_by_role("button", name="Reconstruct", exact=True).click()
    wait_text(page, "Model latency"); time.sleep(3); log.append(f"tab1 {time.time() - t0:.1f} s")
    page.screenshot(path=os.path.join(a.out, "tab1_reconstruct.png"), full_page=True)
    # Tab 2
    page.get_by_role("tab", name="2 · Diagnose & explain").click(); time.sleep(2)
    t0 = time.time(); page.get_by_role("button", name="Compute explanations").click()
    wait_text(page, "Attribution run"); time.sleep(4); log.append(f"tab2 {time.time() - t0:.1f} s")
    page.screenshot(path=os.path.join(a.out, "tab2_diagnose_explain.png"), full_page=True)
    # Tab 3
    page.get_by_role("tab", name="3 · Compare input sets").click(); time.sleep(1)
    t0 = time.time(); page.get_by_role("button", name="Run all input sets (BiMa-SR)").click()
    wait_text(page, "configurations in one call"); time.sleep(4); log.append(f"tab3 {time.time() - t0:.1f} s")
    page.screenshot(path=os.path.join(a.out, "tab3_compare.png"), full_page=True)
    # Tab 4
    page.get_by_role("tab", name="4 · Diffusion uncertainty (lead II)").click(); time.sleep(1)
    t0 = time.time(); page.get_by_role("button", name="Sample", exact=True).click()
    wait_text(page, "draws", timeout=240); wait_text(page, "GPU call"); time.sleep(4)
    log.append(f"tab4 {time.time() - t0:.1f} s")
    page.screenshot(path=os.path.join(a.out, "tab4_uncertainty.png"), full_page=True)
    # Tab 5
    page.get_by_role("tab", name="5 · Results & About").click(); time.sleep(3)
    page.screenshot(path=os.path.join(a.out, "tab5_results_about.png"), full_page=True)
    # Monitor preset -> DiBiMa-SR enabled (Tab 1, small check)
    page.get_by_role("tab", name="1 · Reconstruct").click(); time.sleep(1)
    page.get_by_role("button", name="Monitor").click(); time.sleep(1)
    page.get_by_label("DiBiMa-SR (DDIM-6)").check(); time.sleep(0.5)
    t0 = time.time(); page.get_by_role("button", name="Reconstruct", exact=True).click()
    page.get_by_text("DDIM-6 from II", exact=False).first.wait_for(timeout=180000); time.sleep(3)
    log.append(f"tab1 DDIM-6 {time.time() - t0:.1f} s")
    page.screenshot(path=os.path.join(a.out, "tab1_monitor_ddim6.png"), full_page=True)
    b.close()
print("\n".join(log))
