"""Audit active TeX references against compiled document labels and bibliography."""
from pathlib import Path
import re,json,csv
B=Path(__file__).resolve().parents[1];P=B/'paper';A=B/'outputs/release/reference_audit_20261012';A.mkdir(parents=True,exist_ok=True)
def expand(p):
 s=p.read_text();s=re.sub(r'(?m)(?<!\\)%.*$','',s)
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(p.parent/(m[1]+'.tex')),s)
def labels(stem):
 s=(P/'build'/f'{stem}.aux').read_text()
 return dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',s))
si=labels('supplementary-comparisons');macros={'SIControlledTable':si['tab:controlled'],'SIFinalTable':si['tab:finalstep']}
(P/'supplementary_reference_numbers.tex').write_text('% Generated from compiled Supplementary Information; run the reference audit after SI compilation.\n'+''.join('\\newcommand{\\'+key+'}{'+val+'}\n' for key,val in macros.items()))
main=P/'pair-context-benchmark.tex';s=main.read_text()
if '\\input{supplementary_reference_numbers}' not in s:s=s.replace('\\begin{abstract}','\\input{supplementary_reference_numbers}\n\n\\begin{abstract}')
s=s.replace('Single-task, residual, mean, variance and shared-encoder variants are reported in Supplementary Tables S3--S4.','Validation-selected single-task, residual, mean, variance and shared-encoder variants are reported in Supplementary Table~\\SIControlledTable. Their final-step results are reported in Supplementary Table~\\SIFinalTable.')
s=s.replace('Supplementary Table S4','Supplementary Table~\\SIFinalTable');main.write_text(s)
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',(P/'references.bib').read_text()));rows=[];issues=[]
for stem in ['pair-context-benchmark','supplementary-comparisons']:
 s=expand(P/(stem+'.tex'));ls=labels(stem)
 defs=re.findall(r'\\label\{([^}]+)\}',s)
 for name in set(defs):
  if defs.count(name)>1:issues.append(f'{stem}: duplicate label {name}')
 for typ,key in re.findall(r'\\(ref|eqref|pageref|autoref)\{([^}]+)\}',s):
  rows.append({'document':stem,'type':typ,'label':key,'resolved_number':ls.get(key,'MISSING')})
  if key not in ls:issues.append(f'{stem}: undefined label {key}')
 for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\]){0,2}\{([^}]+)\}',s):
  for key in group.split(','):
   if key.strip() not in keys:issues.append(f'{stem}: missing citation {key}')
 for key,val in macros.items():
  if '\\'+key in s:rows.append({'document':stem,'type':'SI table','label':key,'resolved_number':val})
 if re.search(r'Supplementary Tables?\s+S\d',s):issues.append(f'{stem}: hard-coded supplementary table reference')
with (A/'resolved_references.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['document','type','label','resolved_number']);w.writeheader();w.writerows(rows)
report={'issues':issues,'SI_table_numbers':si,'main_labels':labels('pair-context-benchmark'),'citation_keys_available':len(keys),'fix':'Separate validation-selected variants (S3) from final-step robustness (S4); references generated from SI labels.'}
(A/'report.json').write_text(json.dumps(report,indent=2));assert not issues,issues;print(json.dumps({'issues':issues,'supplementary_tables':{k:v for k,v in si.items() if k.startswith('tab:')}},indent=2))
