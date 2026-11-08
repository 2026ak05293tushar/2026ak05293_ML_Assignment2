"""Create the mandatory single PDF in the specified order.

Without real links and a BITS Virtual Lab screenshot, generates a visibly
INCOMPLETE template. Never fabricates link destinations or proof of execution.

Examples:
 python scripts/build_submission_pdf.py
 python scripts/build_submission_pdf.py --github https://github.com/YOU/REPO \
   --streamlit https://YOUR-APP.streamlit.app/ --screenshot /path/to/REAL-bits-lab.png
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from urllib.parse import urlparse

try:
    import mistune
    from weasyprint import HTML
except ImportError as error:
    raise SystemExit('Install optional PDF dependencies: python -m pip install -r requirements-report.txt') from error

ROOT = Path(__file__).resolve().parents[1]


def as_link(value: str, text: str) -> str:
    if not value:
        return '<span class="missing">NOT PROVIDED - REPLACE WITH REAL LINK</span>'
    u = urlparse(value)
    if u.scheme != 'https' or not u.netloc:
        raise ValueError(f'{text} must be a complete HTTPS URL.')
    return f'<a href="{escape(value, quote=True)}">{escape(value)}</a>'


def create_pdf(github: str, streamlit: str, screenshot: str, output: Path) -> None:
    if screenshot:
        snapshot = Path(screenshot).expanduser().resolve()
        if not snapshot.is_file() or snapshot.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            raise ValueError('--screenshot must be a valid existing PNG or JPEG of real BITS Virtual Lab execution.')
        image_block = f'<img class="lab-image" src="{snapshot.as_uri()}" alt="User-provided BITS Virtual Lab execution screenshot">'
    else:
        image_block = ('<div class="missing-evidence">REQUIRED: INSERT YOUR OWN AUTHENTIC '
                       'BITS VIRTUAL LAB EXECUTION SCREENSHOT HERE.<br>This sample is not proof of BITS execution.</div>')

    not_ready = not all([github, streamlit, screenshot])
    markdown = (ROOT / 'README.md').read_text(encoding='utf-8')
    markdown = markdown.replace('[ADD YOUR ACTUAL GITHUB REPOSITORY URL]', github or '[ADD YOUR ACTUAL GITHUB REPOSITORY URL]')
    markdown = markdown.replace('[ADD YOUR ACTUAL STREAMLIT COMMUNITY CLOUD URL]', streamlit or '[ADD YOUR ACTUAL STREAMLIT COMMUNITY CLOUD URL]')
    markdown = markdown.replace('<!-- RESULTS_FROM_TRAINING_START -->', '').replace('<!-- RESULTS_FROM_TRAINING_END -->', '')
    markdown_html = mistune.create_markdown(plugins=['table'])(markdown)

    # A clear non-final status avoids suggesting placeholder evidence has been verified.
    status_label = 'INCOMPLETE - missing one or more user-owned deliverables' if not_ready else 'EVIDENCE SUPPLIED - verify links and screenshot before submission'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <style>
    @page {{size:A4; margin:17mm 15.5mm 19mm; @bottom-right {{ content:"Page " counter(page) " of " counter(pages); font: 9px Arial, sans-serif; color:#657582; }} }}
    body {{ font-family:DejaVu Sans, Arial, sans-serif; color:#1a2934; font-size:12px; line-height:1.50; }}
    h1 {{ font-size:23px; color:#103b53; margin:0 0 11px; line-height:1.18; }}
    h2 {{ color:#155069; font-size:16px; border-bottom:1px solid #b2ccd6; padding-bottom:5px; margin:18px 0 9px; break-after:avoid; }}
    h3 {{ color:#255870; font-size:13px; margin:14px 0 7px; break-after:avoid; }}
    p, li {{ orphans:3; widows:3; }}
    ol, ul {{ padding-left:18px; }}
    li {{margin-bottom:4px;}}
    a {{color:#0e648b; text-decoration:underline; overflow-wrap:anywhere;}}
    .eyebrow {{font-size:12px; text-transform:uppercase; letter-spacing:1.5px; color:#286580; margin-bottom:13px;}}
    .cover {{margin-top:22mm;}}
    .subtitle {{font-size:13px; color:#5a7180; margin-top:12px;}}
    .status {{margin:18px 0; border:1px solid #c4cbd0; padding:12px; font-size:11px; background:#f6f8fa;}}
    .missing {{color:#9a2323; font-weight:700;}}
    .link-entry {{margin:14px 0; padding:10px 11px; border:1px solid #cddde5;}}
    .link-entry strong {{display:block; margin-bottom:7px;}}
    .lab-image {{max-width:100%; max-height:170mm; object-fit:contain; display:block; margin:8px auto;}}
    .missing-evidence {{min-height:70mm; border:2px dashed #abbcc5; color:#8b3939; background:#fcfcfc; text-align:center; padding:28mm 14mm; box-sizing:border-box; font-weight:bold; line-height:2;}}
    .pagebreak {{page-break-before:always;}}
    table {{border-collapse:collapse; width:100%; font-size:10px; line-height:1.40; margin:8px 0 12px; table-layout:fixed;}}
    th,td {{border:1px solid #cbd8df; padding:5px 5px; overflow-wrap:anywhere; vertical-align:top;}}
    th:first-child, td:first-child {{width:21%;}}
    th {{background:#e9f2f6; color:#173f53; text-align:left;}}
    tr {{break-inside:avoid;}}
    code {{font:10px DejaVu Sans Mono, monospace; word-break:break-word;}}
    pre {{background:#eef3f6; border:1px solid #e0e7ed; padding:8px; white-space:pre-wrap; word-break:break-word; font-size:10px; line-height:1.4;}}
    pre code {{font-size:10px;}}
    blockquote {{border-left:3px solid #4e8ca4; padding:4px 9px; margin-left:0; background:#eef5f8;}}
    </style></head><body>
    <section class="cover">
      <div class="eyebrow">Machine Learning · Assignment 2</div>
      <h1>Five-Model Classification and Streamlit Deployment</h1>
      <p class="subtitle">M.Tech AIML - BITS WILP | UCI Breast Cancer Wisconsin (Diagnostic)</p>
      <p class="status"><b>Submission status:</b> {escape(status_label)}</p>
      <h2>1. GitHub Repository Link</h2><div class="link-entry"><strong>Required: public GitHub repository containing project files</strong>{as_link(github,'GitHub URL')}</div>
      <h2>2. Live Streamlit App Link</h2><div class="link-entry"><strong>Required: deployed interactive app on Community Cloud</strong>{as_link(streamlit,'Streamlit URL')}</div>
      <h2>3. BITS Virtual Lab Screenshot</h2>{image_block}
    </section>
    <section class="pagebreak"><h2>4. Complete README content (included as required)</h2>{markdown_html}</section>
    </body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=page, base_url=str(ROOT)).write_pdf(str(output))
    print('Generated PDF:', output)
    print('Status:', status_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github', default='', help='Your actual public repository HTTPS URL')
    parser.add_argument('--streamlit', default='', help='Your actual deployed Streamlit app HTTPS URL')
    parser.add_argument('--screenshot', default='', help='Genuine BITS Virtual Lab screenshot PNG/JPEG')
    parser.add_argument('--output', default=str(ROOT / 'Assignment_2_Submission.pdf'))
    parser.add_argument('--update-readme', action='store_true',
                        help='Also write the two URLs into README.md (section c) so the GitHub README matches the PDF')
    args = parser.parse_args()
    if args.update_readme:
        if not (args.github.strip() and args.streamlit.strip()):
            raise SystemExit('--update-readme needs both --github and --streamlit.')
        readme = ROOT / 'README.md'
        text = readme.read_text(encoding='utf-8')
        text = text.replace('[ADD YOUR ACTUAL GITHUB REPOSITORY URL]', args.github.strip())
        text = text.replace('[ADD YOUR ACTUAL STREAMLIT COMMUNITY CLOUD URL]', args.streamlit.strip())
        readme.write_text(text, encoding='utf-8')
        print('README.md updated with your links - commit and push it.')
    create_pdf(args.github.strip(), args.streamlit.strip(), args.screenshot.strip(), Path(args.output))


if __name__ == '__main__':
    main()
