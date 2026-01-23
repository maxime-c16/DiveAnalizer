#!/usr/bin/env python3
from bs4 import BeautifulSoup
import subprocess, sys, re

html_path = 'dives/review_gallery.html'
soup = BeautifulSoup(open(html_path, 'r', encoding='utf-8'), 'html.parser')
scripts = [s for s in soup.find_all('script') if not s.get('src')]
print(f'Found {len(scripts)} inline <script> blocks')
failed = False
for i, s in enumerate(scripts):
    code = s.string or ''
    if not code.strip():
        print(f'[{i}] empty, skipped')
        continue
    fn = f'/tmp/inline_script_{i}.js'
    open(fn, 'w', encoding='utf-8').write(code)
    proc = subprocess.run(['node', '--check', fn], capture_output=True, text=True)
    if proc.returncode != 0:
        failed = True
        print(f'[{i}] SYNTAX ERROR')
        print(proc.stderr)
        # print context
        lines = code.splitlines()
        m = re.search(r':(\d+)', proc.stderr)
        ln = int(m.group(1)) if m else None
        if ln:
            start = max(0, ln - 6)
            end = min(len(lines), ln + 4)
            print('--- Context ---')
            for j in range(start, end):
                mark = '->' if j + 1 == ln else '  '
                print(f"{mark} {j+1:4d}: {lines[j]}")
            print('--- End ---')
    else:
        print(f'[{i}] OK')

if failed:
    sys.exit(2)
print('All inline script blocks parse OK')
