"""Offline browser check; run with an isolated Playwright environment."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('html',type=Path);a=p.parse_args()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(); errors=[]; requests=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:requests.append(r.url))
 page.goto(a.html.resolve().as_uri(),wait_until='load',timeout=180000)
 try: page.wait_for_selector('.token',timeout=120000)
 except Exception:
  print('Browser errors:',errors,'UI:',page.locator('#tokens').inner_text())
  raise
 page.locator('.token').first.click()
 assert 'probabilities' in page.locator('#detail').inner_text()
 page.select_option('#color','prediction');page.check('#errors');page.uncheck('#errors')
 for dataset in page.locator('#dataset option').evaluate_all('(xs)=>xs.map(x=>x.value)'):
  page.select_option('#dataset',dataset)
  page.wait_for_selector('.token',timeout=120000)
  for layer in page.locator('#layer option').evaluate_all('(xs)=>xs.map(x=>x.value)'):
   page.select_option('#layer',layer)
   assert 'accuracy' in page.locator('#trajectoryInfo').inner_text()
  largest=page.evaluate('index.map((r,i)=>({...r,i})).filter(r=>r.dataset===document.getElementById("dataset").value).sort((a,b)=>b.tokens-a.tokens)[0].i')
  page.select_option('#trajectory',str(largest))
  page.wait_for_selector('.token',timeout=120000)
  page.locator('.token').first.click()
  assert 'probabilities' in page.locator('#detail').inner_text()
 page.fill('#search','__no_such_trajectory__');assert page.locator('#tokens').inner_text()=='No trajectories'
 assert not errors,errors
 assert all(u.startswith('file:') for u in requests),requests
 print('Offline browser check passed: decompression, token inspection, coloring, filters, no network requests.')
 browser.close()
