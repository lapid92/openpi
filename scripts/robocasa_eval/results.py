"""Validate raw episode evidence and calculate task and group summaries."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

from .protocol import GROUPS, RESET_PROTOCOL, stable_seed


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def validate_episode(row, manifest):
    if row.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("Manifest fingerprint mismatch")
    tasks = {t['task']: t['group'] for t in manifest['tasks']}
    fields = ('scene_seed', 'replan_steps', 'split', 'model') if manifest['model'] == 'smolvla' else ('scene_seed', 'replan_steps')
    for key in fields:
        if row.get(key) != manifest[key]:
            raise ValueError('Protocol field mismatch: ' + key)
    if manifest['model'] == 'smolvla' and row.get('prompt') != row.get('task'):
        raise ValueError('Expected bare task prompt')
    if row.get('scene_reset_protocol') != RESET_PROTOCOL:
        raise ValueError('Wrong scene reset protocol')
    for key in ('campaign_id', 'protocol_id', 'denoising_steps', 'checkpoint_revision'):
        if row.get(key) != manifest[key]:
            raise ValueError('Episode identity mismatch: '+key)
    task = row.get('task')
    if task not in tasks or row.get('group') != tasks[task]:
        raise ValueError('Unknown task/group')
    idx = row.get('episode_index')
    if type(idx) is not int or not 0 <= idx < manifest['episodes_per_task']:
        raise ValueError('Episode index outside manifest')
    if row.get('status') not in ('success', 'policy_failure', 'infrastructure_failure'):
        raise ValueError('Nonterminal episode')
    if row['status'] != 'infrastructure_failure':
        tag = 'public_smolvla_baseline' if manifest['model'] == 'smolvla' else 'public_pi05_baseline'
        if row.get('policy_seed') != stable_seed(tag, task, idx, manifest['policy_seed']):
            raise ValueError('Policy seed mismatch')
        if not all(row.get(k) for k in ('model_xml_sha256', 'initial_state_sha256')):
            raise ValueError('Missing scene hashes')
        if type(row.get('policy_queries')) is not int or row['policy_queries'] <= 0:
            raise ValueError('Missing policy queries')
        if type(row.get('success')) is not bool or row['success'] != (row['status'] == 'success'):
            raise ValueError('Inconsistent success flag')
        if row.get('action_network_evaluations') != row.get('policy_queries', -1)*manifest['denoising_steps']:
            raise ValueError('Unverified denoising count')
    return task, idx


def summarize(rows, manifest):
    by_key = {}
    for row in rows:
        key = validate_episode(row, manifest)
        if key in by_key:
            # Distinct attempts must never silently replace a previous result.
            if row != by_key[key]: raise ValueError('Conflicting duplicate episode: '+str(key))
            continue
        by_key[key] = row
    def stats(task_names, label, reference=None):
        selected=[r for (task,_),r in by_key.items() if task in task_names]
        completed=[r for r in selected if r['status'] in ('success','policy_failure')]
        errors=sum(r['status']=='infrastructure_failure' for r in selected)
        expected=len(task_names)*manifest['episodes_per_task']
        n=len(completed);wins=sum(r['success'] for r in completed)
        complete=n==expected and not errors
        rate=100*wins/n if n else None
        return dict(name=label,expected_episodes=expected,completed_episodes=n,successes=wins,
                    infrastructure_errors=errors,missing_episodes=expected-len(selected),
                    coverage_complete=complete,success_rate_pct=rate,
                    public_reference_pct=reference,
                    delta_percentage_points=rate-reference if complete and rate is not None and reference is not None else None,
                    comparison_status='complete' if complete else 'incomplete_do_not_compare')
    per_task=[dict(stats({t['task']},t['task']), group=t['group']) for t in manifest['tasks']]
    groups=[stats({t['task'] for t in manifest['tasks'] if t['group']==g},g,manifest['reference']['group_rates_pct'][g]) for g in GROUPS]
    refs=[manifest['reference']['group_rates_pct'][t['group']] for t in manifest['tasks']]
    weighted=sum(refs)/len(refs) if refs and all(x is not None for x in refs) else None
    groups.append(stats({t['task'] for t in manifest['tasks']},'overall',manifest['reference'].get('overall_rate_pct',weighted)))
    return list(by_key.values()),per_task,groups


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    tmp.replace(path)


def save_tables(rows, manifest, out):
    episodes,tasks,groups=summarize(rows,manifest)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'episodes.csv',episodes)
    write_csv(out/'per_task.csv',tasks)
    write_csv(out/'comparison.csv',groups)
    (out/'coverage.json').write_text(json.dumps(groups[-1],indent=2))
    return groups[-1]
