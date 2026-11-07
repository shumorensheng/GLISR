"""Certified epoch6 continuation of the unchanged supervised reference."""
import argparse
import json
from pathlib import Path
import resource
import sys

from runs import shared_supervised_reference as original
from runs import shared_supervised_json_readout as reader

ROOT, BASE = original.ROOT, original.BASE
PARENT = BASE / 'shared_supervised_reference_v1'
RUN = BASE / 'shared_supervised_reference_json_resume_v1'
PARENT_ENTITY_BYTES = 3869851945
original.RUN = RUN
load_protocol = original.load_protocol
configure_runtime = original.configure_runtime
digest = original.digest
atomic_torch = original.atomic_torch
atomic_predictions = original.atomic_predictions
write_json = original.write_json


def effective_addendum():
    return json.loads((RUN/'resume_addendum.json').read_text())


def resources():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = original.torch.cuda.max_memory_reserved(0) if original.torch.cuda.is_initialized() else 0
    own = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
    joint = PARENT_ENTITY_BYTES + own
    assert rss <= 32*1024**3 and gpu <= 16*1024**3 and joint <= 16*1024**3, (rss,gpu,joint)
    return {'RSS_peak_bytes': rss, 'GPU_peak_reserved_bytes': gpu, 'new_entity_bytes': own,
            'read_only_parent_entity_bytes': PARENT_ENTITY_BYTES, 'joint_disk_bytes': joint}


original.resources = resources
reader.install_serialization_fix()


def execute(phase,group,state=None,task=None):
    if phase in ('extract','fit'):
        started = original.time.monotonic()
        admission = {'phase':'GPU_entry','group':group,'state':state,'task':task,'passed':None}
        try:
            admission['passed'] = original.gpu_entry(f'{phase}:{state}:{task}')
        finally:
            admission['seconds'] = original.time.monotonic()-started
            with (RUN/'compute_ledger.jsonl').open('a') as output: output.write(json.dumps(admission)+'\n')
        if not admission['passed']:
            write_json(RUN/'deferred.json',{'status':'DEFERRED_AT_GPU_BOUNDARY_NO_QUEUE','phase':phase,'admission':admission})
            return False
    allowed = original.remaining_seconds(group)
    assert allowed > 0
    command=[sys.executable,'-m','runs.shared_supervised_json_resume',phase]
    if state is not None:command+=['--state',state]
    if task is not None:command+=['--task',task]
    log=RUN/('_'.join(str(x) for x in (phase,state,task) if x is not None)+'.log')
    started=original.time.monotonic()
    with log.open('w') as output:
        process=original.subprocess.Popen(command,stdout=output,stderr=original.subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'phase':phase,'state':state,'task':task,'pid':process.pid,'log':str(log),'seconds_available':allowed}),flush=True)
        rss=0
        while process.poll() is None:
            if original.time.monotonic()-started>allowed:
                original.os.killpg(process.pid,original.signal.SIGTERM);process.wait();break
            status=Path(f'/proc/{process.pid}/status')
            if status.exists():
                fields={line.split(':')[0]:line.split(':',1)[1].strip() for line in status.read_text().splitlines() if ':' in line}
                if 'VmHWM' in fields:
                    rss=max(rss,int(fields['VmHWM'].split()[0])*1024)
                    if rss+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>32*1024**3:
                        original.os.killpg(process.pid,original.signal.SIGTERM);process.wait();break
            original.time.sleep(2)
        code=process.wait()
    row={'phase':phase,'group':group,'state':state,'task':task,'seconds':original.time.monotonic()-started,
         'pid':process.pid,'exit_code':code,'RSS_peak_observed_bytes':rss,'closed_log_sha256':digest(log)}
    with (RUN/'compute_ledger.jsonl').open('a') as output:output.write(json.dumps(row)+'\n')
    if code!=0:
        write_json(RUN/'failure_closure.json',{'status':'FAILED_PHASE_STOP_NO_RETRY','phase':row,'ledger':original.ledger()})
        return False
    assert row['seconds']<=allowed and sum(x['seconds'] for x in original.ledger())<=129600
    return True


def close():
    original.close()
    addendum=effective_addendum()
    protocol=load_protocol()
    import_source=RUN/'readouts/T0/fsod/imported_epoch5_certificate.json'
    certificate=json.loads(import_source.read_text())
    assert certificate['completed_updates']==1475 and certificate['next_update']==1476
    assert certificate['epoch']==5 and certificate['next_epoch']==6
    report=json.loads((RUN/'readouts/T0/fsod/report.json').read_text())
    assert report['updates']==17700 and report['epochs']==60
    write_json(RUN/'resume_completion.json',{'status':'SERIALIZATION_CONTINUATION_COMPLETE_ORIGINAL_FAILURE_RETAINED',
        'scientific_protocol_sha256':digest(RUN/'protocol.json'),
        'resume_addendum_sha256':digest(RUN/'resume_addendum.json'),
        'repair_receipt_sha256':addendum['repair_receipt_sha256'],
        'imported_epoch5_certificate_sha256':digest(import_source),
        'original_T0_FSOD_updates':1475,'new_T0_FSOD_updates':16225,
        'T0_FSOD_logical_updates':17700,'T0_FSOD_physical_updates':17700,
        'all_eight_target_heads_logical_updates':81600,'all_eight_target_heads_physical_updates':81600,
        'source_body_updates_unchanged':18000,'no_epoch5_update_or_validation_replayed':True,
        'reconstructed_epoch5_log':addendum['reconstructed_epoch5_log'],
        'parent_failure_sha256':addendum['parent_failure_sha256'],
        'metadata_copy_forensic_import_flag_preserved':True,'no_CUDA_bitwise_reproduction_claim':True,
        'resources':resources()})
    completed=json.loads((RUN/'completion.json').read_text())
    completed['resume_completion_sha256']=digest(RUN/'resume_completion.json')
    completed['resume_addendum_sha256']=digest(RUN/'resume_addendum.json')
    completed['inherited_failure_preserved']=True
    write_json(RUN/'completion.json',completed)


def pipeline():
    addendum=effective_addendum()
    for path,expected in addendum['preforward_bindings_sha256'].items():assert digest(path)==expected,path
    assert digest(RUN/'protocol.json')==addendum['scientific_protocol_sha256']
    assert addendum['parent_entity_bytes']==PARENT_ENTITY_BYTES
    if not execute('fit','fsod:T0','T0','fsod'):return
    if not execute('score','score','T0','fsod'):return
    for state in original.STATES:
        for task in original.TASKS:
            if state=='T0' and task=='fsod':continue
            if not execute('extract','encode',state,task):return
            if not execute('fit',task+':'+state,state,task):return
            if not execute('score','score',state,task):return
    if not execute('close','score'):return
    rows=original.ledger()
    closed=json.loads((RUN/'completion.json').read_text())
    closed['ledger_final_seconds']=sum(row['seconds'] for row in rows)
    closed['new_formal_phase_exits']=[{k:row[k] for k in ('phase','state','task','pid','exit_code')}
                                     for row in rows[addendum['preforward_ledger_rows']:] if 'pid' in row]
    closed['inherited_failure_preserved']=True
    write_json(RUN/'completion.json',closed)
    print(json.dumps({'status':closed['status'],'completion_sha256':digest(RUN/'completion.json')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('fit','extract','score','close','pipeline'))
    parser.add_argument('--state');parser.add_argument('--task')
    args=parser.parse_args()
    if args.phase=='pipeline':pipeline()
    elif args.phase=='close':close()
    else:
        {'fit':reader.fit_head,'extract':reader.targetextract,'score':reader.score_readout}[args.phase](args.state,args.task)
