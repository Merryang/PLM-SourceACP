"""Inspect or rerun an archived training configuration with portable paths."""
from pathlib import Path
import argparse,ast,json,subprocess,sys,shlex
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--execute',action='store_true');a=p.parse_args()
 cfg=json.loads(a.config.read_text(encoding='utf-8'))
 script='external_transfer.py' if 'mode' in cfg else 'train_plm_metaacp.py'
 tree=ast.parse((ROOT/'meta_acp'/script).read_text(encoding='utf-8'));flags={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant):
   flag=n.args[0].value
   if isinstance(flag,str) and flag.startswith('--'):flags[flag[2:].replace('-','_')]=flag
 cfg['device']=a.device
 original=Path(str(cfg['out_dir']).replace('\\','/'))
 if original.is_absolute():raise ValueError('Non-portable archived output path')
 cfg['out_dir']=str(Path('retrained')/original)
 cmd=[sys.executable,'-m','meta_acp.'+Path(script).stem]
 for k,v in cfg.items():
  if k not in flags or v is None or v is False:continue
  cmd.append(flags[k])
  if v is True:continue
  values=v if isinstance(v,list) else [v]
  cmd.extend(str(z).replace('\\','/') for z in values)
 print(shlex.join(cmd))
 if a.execute:
  for k in ['train','validation','test','embedding_cache']:
   if cfg.get(k) and not (ROOT/str(cfg[k]).replace('\\','/')).exists():raise FileNotFoundError(f'{k}: {cfg[k]}')
  subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
