"""Bounded local queue for the frozen 84-job GLISR supplement."""
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import time
import traceback

ROOT = Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))
RUN = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))/'glisr_supplement_20261104_v1'
PYTHON = sys.executable


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prefix_path(job):
    name = 'counted_source_prefix50.json' if job['phase'] == 'source' else 'counted_target_prefix50.json'
    return RUN/job['id']/name


def command(job, deadline):
    common = ['--seed', str(job['seed']), '--deadline', str(deadline)]
    if job['phase'] == 'source':
        return [PYTHON, '-B', '-u', '-m', 'runs.glisr_supplement_variants', 'source', '--variant', job['variant'], *common]
    if job['phase'] == 'target':
        return [PYTHON, '-B', '-u', '-m', 'runs.glisr_supplement_targets', 'target', '--variant', job['variant'], '--task', job['task'], *common]
    ready = json.loads((RUN/'cfc_worker_ready.json').read_text())
    return [PYTHON, '-B', '-u', '-m', ready['module'], 'worker', '--method', job['method'], *common]


def pipeline():
    p = json.loads((RUN/'protocol.json').read_text())
    r = p['resources']
    jobs = p['jobs']
    assert len(jobs) == 84
    assert not (RUN/'queue_activation.json').exists(), 'Queue already activated; inspect existing actual workers before a new attempt'
    started = time.time()
    deadline = started+r['global_hard_seconds']
    write(RUN/'queue_activation.json', {'supervisor_PID': os.getpid(), 'started_unix': started,
        'deadline_unix': deadline, 'goal_unix': started+24*3600, 'physical_GPUs': r['physical_GPUs']})
    (RUN/'queue_snapshot.py').write_text(Path(__file__).read_text())
    for name in ('glisr_supplement_variants.py', 'glisr_supplement_targets.py'):
        (RUN/('launch_'+name)).write_text((ROOT/'runs'/name).read_text())
    pending = list(jobs)
    active = {}
    statuses = {j['id']: 'PENDING' for j in jobs}
    waits, launches = [], {}
    cpus = sorted(os.sched_getaffinity(0))
    try:
        while pending or active:
            now = time.time()
            if now > deadline:
                raise TimeoutError('Frozen supplement96h global deadline')
            for jid, record in list(active.items()):
                child, log, job = record['child'], record['log'], record['job']
                if now > record['deadline_unix'] and child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait()
                log.close()
                del active[jid]
                status = 'COMPLETE' if code == 0 else 'FAILED'
                expected = RUN/jid/('source_report.json' if job['phase'] == 'source' else 'results.json')
                if code == 0:
                    report = json.loads(expected.read_text())
                    success = report['status'] == ('COMPLETE_SOURCE6000' if job['phase'] == 'source' else 'COMPLETE')
                    if not success:
                        status = 'FAILED'
                statuses[jid] = status
                receipt = {'job': jid, 'PID': child.pid, 'GPU': record['GPU'],
                    'actual_wait_exit': code, 'status': status, 'wall_seconds': now-record['started_unix']}
                waits.append(receipt)
                write(RUN/jid/'actual_wait.json', receipt)
                write(RUN/'actual_waits.json', waits)
            for job in list(pending):
                dependency = job['dependency']
                if dependency is not None and statuses[dependency] in ('FAILED', 'BLOCKED'):
                    statuses[job['id']] = 'BLOCKED'
                    write(RUN/job['id']/'blocked.json', {'status': 'BLOCKED', 'dependency': dependency, 'dependency_status': statuses[dependency]})
                    pending.remove(job)
            target_ready = (RUN/'target_worker_ready.json').is_file()
            cfc_ready = (RUN/'cfc_worker_ready.json').is_file() and (RUN/'cfc_actual_check.json').is_file()
            # GPU3 remains free for the separately reviewed real CFC check.
            for gpu in r['physical_GPUs']:
                own = [v for v in active.values() if v['GPU'] == gpu]
                if gpu == 3 and not cfc_ready:
                    continue
                if len(own) >= r['max_owned_workers_per_GPU']:
                    continue
                if any(not prefix_path(v['job']).is_file() for v in own):
                    continue
                eligible = [j for j in pending if
                    (j['dependency'] is None or statuses[j['dependency']] == 'COMPLETE') and
                    (j['phase'] == 'source' or j['phase'] == 'target' and target_ready or j['phase'] == 'cfc' and cfc_ready)]
                # B1's first counted prefix runs alone on the reserved GPU3.
                b1_prefix = RUN/'B1_s0_source/counted_source_prefix50.json'
                if not b1_prefix.is_file():
                    if gpu == 3:
                        eligible = [j for j in eligible if j['id'] == 'B1_s0_source'] if not own else []
                    else:
                        eligible = [j for j in eligible if j.get('variant') != 'B1']
                if not eligible:
                    continue
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free < r['free_admission_MiB']:
                    continue
                eligible.sort(key=lambda j: 0 if j['phase'] == 'target' else 1 if j['phase'] == 'source' else 2)
                job = eligible[0]
                cap = r['source_seconds'] if job['phase'] == 'source' else r['target_seconds'][job['task']]
                limit = min(deadline, now+cap)
                folder = RUN/job['id']
                argv = command(job, limit)
                write(folder/'job.json', {**job, 'GPU': gpu, 'deadline_unix': limit, 'admission_free_MiB': free, 'argv': argv})
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
                log = (folder/'stdout.log').open('w')
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                os.sched_setaffinity(child.pid, {cpus[(4*len(launches)+i) % len(cpus)] for i in range(min(4, len(cpus)))})
                pending.remove(job)
                statuses[job['id']] = 'RUNNING'
                record = {'child': child, 'log': log, 'job': job, 'GPU': gpu, 'deadline_unix': limit, 'started_unix': time.time()}
                active[job['id']] = record
                launches[job['id']] = {'PID': child.pid, 'GPU': gpu, 'started_unix': record['started_unix'], 'deadline_unix': limit, 'argv': argv}
                write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'status': 'RUNNING', 'workers': launches})
            write(RUN/'queue_state.json', {'status': 'RUNNING' if active else 'WAITING_READY_OR_RESOURCE',
                'supervisor_PID': os.getpid(), 'updated_unix': time.time(), 'deadline_unix': deadline,
                'active': {jid: {'PID': v['child'].pid, 'GPU': v['GPU'], 'deadline_unix': v['deadline_unix']} for jid, v in active.items()},
                'pending': [j['id'] for j in pending], 'statuses': statuses,
                'gates': {'target_worker_ready': target_ready, 'cfc_worker_and_actual_check_ready': cfc_ready}})
            time.sleep(3)
        write(RUN/'queue_completion.json', {'status': 'COMPLETE' if all(v == 'COMPLETE' for v in statuses.values()) else 'CLOSED_WITH_FAILURES',
            'statuses': statuses, 'actual_waits': waits, 'wall_seconds': time.time()-started})
    except BaseException:
        write(RUN/'queue_failure.json', {'traceback': traceback.format_exc(), 'time_unix': time.time()})
        raise
    finally:
        for jid, v in active.items():
            child = v['child']
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            v['log'].close()
            waits.append({'job': jid, 'PID': child.pid, 'GPU': v['GPU'], 'actual_wait_exit': code, 'cleanup': True})
        write(RUN/'actual_waits.json', waits)


def cfc_worker(method, seed, deadline):
    # The running supervisor delegates through this entrypoint after review gates.
    folder = RUN/f'CFC26_{method}_s{seed}'
    import fcntl
    with (folder/'execution_assignment.lock').open('a') as assignment_lock:
        fcntl.flock(assignment_lock, fcntl.LOCK_EX)
        delegated = (folder/'remote_assignment.json').is_file()
        joining_local = False
        if delegated:
            assignment = json.loads((folder/'remote_assignment.json').read_text())
            deadline = min(deadline, assignment['deadline_unix'])
        else:
            owner_pids = []
            for name in ('local_launch_reserved.json', 'train_launch.json'):
                marker = folder/name
                if marker.is_file():
                    pid = json.loads(marker.read_text())['PID']
                    state = Path(f'/proc/{pid}/stat')
                    if state.exists() and state.read_text().split()[2] != 'Z':
                        owner_pids.append(pid)
            joining_local = bool(owner_pids)
            if not joining_local:
                write(folder/'local_launch_reserved.json', {'PID': os.getpid(), 'reserved_unix': time.time(),
                    'deadline_unix': deadline, 'hostname': os.uname().nodename})
    if joining_local:
        write(folder/'local_join.json', {'PID': os.getpid(), 'role': 'JOIN_EXISTING_LOCAL_NO_GPU_WORK',
            'owner_PIDs': owner_pids, 'deadline_unix': deadline})
        while not (folder/'results.json').is_file():
            if time.time() > deadline:
                raise TimeoutError('Existing local CFC join deadline')
            alive = [pid for pid in owner_pids if Path(f'/proc/{pid}/stat').exists()
                and Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z']
            if not alive:
                raise RuntimeError('Existing local CFC execution ended without complete results')
            time.sleep(3)
        assert json.loads((folder/'results.json').read_text())['status'] == 'COMPLETE'
        return
    if delegated:
        write(folder/'remote_join.json', {'PID': os.getpid(), 'role': 'JOIN_ONLY_NO_LOCAL_GPU_WORK',
            'assignment': str(folder/'remote_assignment.json'), 'deadline_unix': deadline})
        while not (folder/'remote_actual_wait.json').is_file():
            if time.time() > deadline:
                raise TimeoutError('Delegated CFC original deadline')
            time.sleep(3)
        waited = json.loads((folder/'remote_actual_wait.json').read_text())
        if waited['actual_wait_exit'] != 0:
            raise RuntimeError('Delegated CFC failed: '+str(waited))
        assert json.loads((folder/'completion.json').read_text())['status'] == 'COMPLETE_CFC26_TERMINAL_TEST'
        assert json.loads((folder/'results.json').read_text())['status'] == 'COMPLETE'
        return
    prefix = folder/'counted_target_prefix50.json'
    if not prefix.is_symlink():
        prefix.symlink_to('counted_prefix50.json')
    (folder/'queue_adapter_snapshot.py').write_text(Path(__file__).read_text())
    # Serialize admission through the actual first50 counted updates per GPU.
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    admission = (RUN/f'gpu{gpu}_cfc_admission.lock').open('a')
    fcntl.flock(admission, fcntl.LOCK_EX)
    dependency = RUN.parent/'featsharp_marine_matched_v1/build_dependencies'
    env = dict(os.environ, PATH=str(dependency/'bin')+':'+os.environ['PATH'],
        PYTHONPATH=str(dependency)+':'+str(ROOT), MAX_JOBS='4')
    peaks = []
    for peer in RUN.glob(f'CFC26_{method}_s*/counted_prefix50.json'):
        peaks.append(json.loads(peer.read_text())['resources']['GPU_reserved_bytes'])
    required_MiB = max(8192, int(max(peaks, default=0)*1.25/1024**2)+2048)
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        peers = []
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                argv = (process/'cmdline').read_bytes().split(b'\0')
                if b'runs.glisr_cfc26_train' not in argv or b'train' not in argv:
                    continue
                environ = (process/'environ').read_bytes().split(b'\0')
            except (FileNotFoundError, ProcessLookupError):
                continue
            if ('CUDA_VISIBLE_DEVICES='+gpu).encode() not in environ:
                continue
            peer_method = argv[argv.index(b'--method')+1].decode()
            peer_seed = argv[argv.index(b'--seed')+1].decode()
            peer_folder = RUN/f'CFC26_{peer_method}_s{peer_seed}'
            log = peer_folder/'training.jsonl'
            if not log.is_file() or not log.stat().st_size:
                continue
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            latest = rows[-1]
            start = next((row for row in reversed(rows) if row['updates'] <= latest['updates']-100), rows[0])
            prefix50 = peer_folder/'counted_prefix50.json'
            if latest['updates'] == start['updates']:
                if not prefix50.is_file():
                    continue
                prefix_row = json.loads(prefix50.read_text())
                rate = prefix_row['training_seconds']/50
                update = 50
            else:
                rate = (latest['training_seconds']-start['training_seconds'])/(latest['updates']-start['updates'])
                update = latest['updates']
            evaluations = [json.loads(path.read_text())['seconds'] for path in peer_folder.glob('val_epoch*_metrics.json')]
            peers.append({'job': peer_folder.name, 'PID': int(process.name), 'updates': update,
                'seconds_per_update': rate, 'deadline_unix': float(argv[argv.index(b'--deadline')+1]),
                'remaining_evaluation_seconds': max(evaluations, default=0)*(14-len(evaluations))})
        factor = (len(peers)+1)/len(peers) if peers else 1
        for peer in peers:
            peer['projected_remaining_seconds_with_added_context'] = ((52500-peer['updates'])*peer['seconds_per_update']+peer['remaining_evaluation_seconds'])*factor
            peer['remaining_deadline_seconds'] = peer['deadline_unix']-time.time()
        budget_fits = all(peer['projected_remaining_seconds_with_added_context'] < peer['remaining_deadline_seconds'] for peer in peers)
        write(folder/'admission_forecast.json', {'GPU': gpu, 'checked_unix': time.time(), 'free_MiB': free,
            'required_worker_MiB': required_MiB, 'leave_free_MiB': 8192, 'existing_jobs': peers,
            'additional_context_factor': factor, 'existing_deadlines_fit': budget_fits,
            'method': 'Latest100 normal counted updates plus observed validation/test allowance; linear additional-context projection, not a benchmark'})
        if free >= required_MiB+8192 and budget_fits:
            break
        if time.time() > deadline:
            raise TimeoutError('CFC actual free-memory admission deadline')
        time.sleep(3)
    write(folder/'memory_admission.json', {'GPU': gpu, 'actual_free_MiB': free,
        'method_peer_peak_bytes': peaks, 'required_worker_MiB': required_MiB,
        'leave_free_MiB': 8192, 'first50_serialized': True, 'admitted_unix': time.time()})
    for phase in ('train', 'score'):
        argv = [PYTHON, '-B', '-u', '-m', 'runs.glisr_cfc26_train', phase,
            '--method', method, '--seed', str(seed), '--deadline', str(deadline)]
        child = subprocess.Popen(argv, cwd=ROOT, env=env)
        write(folder/(phase+'_launch.json'), {'PID': child.pid, 'GPU': gpu, 'argv': argv, 'started_unix': time.time()})
        if phase == 'train':
            while not (folder/'counted_prefix50.json').is_file() and child.poll() is None:
                if time.time() > deadline:
                    os.kill(child.pid, signal.SIGKILL)
                    break
                time.sleep(3)
            fcntl.flock(admission, fcntl.LOCK_UN)
            admission.close()
        code = child.wait()
        write(folder/(phase+'_actual_wait.json'), {'PID': child.pid, 'actual_wait_exit': code})
        if code:
            raise RuntimeError(f'CFC {phase} exited{code}')
    completion = json.loads((folder/'completion.json').read_text())
    assert completion['status'] == 'COMPLETE_CFC26_TERMINAL_TEST'
    write(folder/'results.json', {'status': 'COMPLETE', 'native_completion': completion})


def repair_waiting(attempt_name='engineering_wait_fix_v1'):
    import shutil
    attempt = RUN/attempt_name
    request = json.loads((attempt/'request.json').read_text())
    active, receipts = {}, []
    for item in request:
        jid, old_pid = item['job'], item['old_PID']
        folder, evidence = RUN/jid, attempt/jid
        evidence.mkdir(exist_ok=True)
        for name in ('stdout.log', 'config.json', 'queue_adapter_snapshot.py', 'train_actual_wait.json', 'train_launch.json', 'training.jsonl'):
            path = folder/name
            if path.is_file():
                shutil.copy2(path, evidence/name)
        if old_pid is not None:
            children = Path(f'/proc/{old_pid}/task/{old_pid}/children').read_text().strip()
            assert not children, (jid, children)
            argv = Path(f'/proc/{old_pid}/cmdline').read_bytes().split(b'\0')
            assert b'runs.glisr_supplement_queue' in argv and b'worker' in argv
            os.killpg(old_pid, signal.SIGTERM)
            old_state = Path(f'/proc/{old_pid}/stat')
            while old_state.exists() and old_state.read_text().split()[2] != 'Z':
                time.sleep(.05)
            previous_attempt = folder/'engineering_replacement.json'
            if previous_attempt.is_file():
                previous = json.loads(previous_attempt.read_text())
                while previous['replacement_PID'] == old_pid and previous['status'] == 'ENGINEERING_REPLACEMENT_RUNNING':
                    time.sleep(.1)
                    previous = json.loads(previous_attempt.read_text())
        write(evidence/'engineering_attempt.json', {**item, 'reason': 'Reload childless wrapper for recorded engineering resource decision; original deadline and scientific settings retained',
            'termination_requested_unix': time.time(), 'original_actual_wait': str(folder/'actual_wait.json'), 'counted_training_updates_in_old_waiter': 0})
        argv = [PYTHON, '-B', '-u', '-m', 'runs.glisr_supplement_queue', 'worker', '--method', item['method'],
                '--seed', str(item['seed']), '--deadline', str(item['deadline_unix'])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(item['GPU']), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        log = (evidence/'replacement_stdout.log').open('w')
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[jid] = {'child': child, 'log': log, 'item': item}
        receipt = {**item, 'replacement_PID': child.pid, 'status': 'ENGINEERING_REPLACEMENT_RUNNING', 'argv': argv}
        receipts.append(receipt)
        write(folder/'engineering_replacement.json', receipt)
        write(attempt/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'jobs': receipts})
    while active:
        for jid, record in list(active.items()):
            child, item = record['child'], record['item']
            if time.time() > item['deadline_unix'] and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            if child.poll() is None:
                continue
            code = child.wait()
            record['log'].close()
            receipt = {**item, 'replacement_PID': child.pid, 'actual_wait_exit': code,
                'status': 'COMPLETE' if code == 0 else 'FAILED', 'finished_unix': time.time()}
            write(attempt/jid/'replacement_actual_wait.json', receipt)
            write(RUN/jid/'engineering_replacement.json', receipt)
            del active[jid]
        canonical = json.loads((RUN/'queue_state.json').read_text())
        canonical['canonical_state'] = str(RUN/'queue_state.json')
        canonical['engineering_overlay'] = str(attempt/'launch_receipt.json')
        for item in request:
            jid = item['job']
            replacement = json.loads((RUN/jid/'engineering_replacement.json').read_text())
            canonical['statuses'][jid] = 'RUNNING' if jid in active else replacement['status']
            canonical['active'].pop(jid, None)
            if jid in active:
                canonical['active'][jid] = {'PID': active[jid]['child'].pid, 'GPU': item['GPU'], 'deadline_unix': item['deadline_unix'], 'engineering_replacement': True}
        canonical['updated_unix'] = time.time()
        if attempt_name == 'engineering_wait_fix_v1':
            write(RUN/'effective_queue_state.json', canonical)
        else:
            for receipt_path in RUN.glob('*/engineering_replacement.json'):
                receipt = json.loads(receipt_path.read_text())
                canonical['statuses'][receipt['job']] = 'RUNNING' if receipt['status'] == 'ENGINEERING_REPLACEMENT_RUNNING' else receipt['status']
            write(attempt/'state.json', canonical)
        time.sleep(3)


def resume_worker(method, seed, deadline):
    """Observe restore/first normal updates while running the unchanged --resume CLI."""
    import runpy
    import sys
    import random
    import numpy as np
    import torch
    folder = RUN/f'CFC26_{method}_s{seed}'
    saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
    for name in ('model', 'head', 'progress', 'optimizers', 'shuffle_rng', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'):
        saved[name]
    expected_updates = saved['progress']['updates']
    expected_progress = dict(saved['progress'])
    native = ROOT/'runs/glisr_cfc26_train.py'
    lines = native.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('def train('))
    restore_line = next(i+1 for i in range(start, len(lines)) if lines[i] == "    rows = tracking.load_rows('train')")
    restored = False
    first = False
    initial_training_seconds = saved['progress']['training_seconds']
    before = None

    def equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            return torch.equal(actual.detach().cpu(), expected)
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(equal(actual[k], v) for k, v in expected.items())
        if isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
        if isinstance(expected, np.ndarray):
            return np.array_equal(actual, expected)
        return actual == expected

    def observe(frame, event, arg):
        nonlocal restored, first, before
        if frame.f_code.co_filename != str(native) or frame.f_code.co_name != 'train':
            return None
        if event != 'line':
            return observe
        local = frame.f_locals
        if frame.f_lineno == restore_line and not restored:
            proof = {'model_strict_state_equal': equal(local['model'].state_dict(), saved['model']),
                'head_strict_state_equal': equal(local['head'].state_dict(), saved['head']),
                'optimizer_states_equal': equal([o.state_dict() for o in local['optimizers']], saved['optimizers']),
                'progress_order_cursor_equal': equal(local['state'], expected_progress),
                'private_shuffle_rng_equal': torch.equal(local['generator'].get_state(), saved['shuffle_rng']),
                'CPU_rng_equal': torch.equal(torch.get_rng_state(), saved['torch_rng']),
                'CUDA_rng_equal': equal(torch.cuda.get_rng_state_all(), saved['cuda_rng']),
                'numpy_rng_equal': equal(np.random.get_state(), saved['numpy_rng']),
                'python_rng_equal': random.getstate() == saved['python_rng']}
            assert all(proof.values()), proof
            before = local['model'].encoder.backbone.blocks[0].attn.qkv.weight.detach().clone()
            write(folder/'actual_resume_binding.json', {'status': 'FULL_STATE_RESTORED', 'saved_updates': expected_updates,
                'proof': proof, 'PID': os.getpid(), 'hostname': os.uname().nodename,
                'torch': torch.__version__, 'CUDA': torch.version.cuda, 'GPU': torch.cuda.get_device_name(0),
                'claim': 'Saved-state restoration across hardware/software versions; no bitwise trajectory equivalence claimed'})
            restored = True
        if restored and local['state']['updates'] == expected_updates+1 and not first:
            moved = not torch.equal(before, local['model'].encoder.backbone.blocks[0].attn.qkv.weight)
            assert moved
            write(folder/'first_resumed_update.json', {'status': 'PASS_NORMAL_RESUMED_UPDATE',
                'saved_updates': expected_updates, 'updates': local['state']['updates'], 'actual_indices': local['indices'],
                'loss': float(local['loss'].detach()), 'gradient_norms': local['gradient_norms'],
                'backbone_moved': moved, 'input_shape': list(local['x'].shape),
                'PID': os.getpid(), 'GPU': os.environ['CUDA_VISIBLE_DEVICES']})
            first = True
        if restored and local['state']['updates'] == expected_updates+50:
            write(folder/'counted_resume_prefix50.json', {'updates': local['state']['updates'], 'resumed_updates': 50,
                'training_seconds': local['state']['training_seconds']-initial_training_seconds,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved(), 'PID': os.getpid()})
            sys.settrace(None)
            return None
        return observe

    sys.argv = [str(native), 'train', '--method', method, '--seed', str(seed), '--deadline', str(deadline), '--resume']
    sys.settrace(observe)
    runpy.run_module('runs.glisr_cfc26_train', run_name='__main__')
    sys.settrace(None)


def migrate_remote():
    import fcntl
    import shutil
    stop = json.loads((RUN/'remote_3090_stop_receipt.json').read_text())
    assert stop['status'] == 'STOPPED_ALL_THREE_CHECKPOINTS_TRANSFERRED' and stop['gpu_contexts_confirmed_absent']
    attempt = RUN/'gpu0_remote_migration'
    attempt.mkdir(exist_ok=True)
    active = {}
    for item in stop['jobs']:
        jid = item['job']
        folder = RUN/jid
        method = jid[len('CFC26_'):].rsplit('_s', 1)[0]
        with (folder/'execution_assignment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not (folder/'migration_owner.json').exists()
            assignment = json.loads((folder/'remote_assignment.json').read_text())
            deadline = assignment['deadline_unix']
            snapshot = folder/'remote_3090_snapshot'
            assert (snapshot/'last.pt').stat().st_size == item['checkpoint_bytes']
            shutil.copy2(snapshot/'last.pt', folder/'last.pt')
            for path in snapshot.glob('val_epoch*'):
                shutil.copy2(path, folder/path.name)
            if (snapshot/'best.pt').is_file():
                shutil.copy2(snapshot/'best.pt', folder/'best.pt')
            rows = [json.loads(line) for line in (snapshot/'training.jsonl').read_text().splitlines()]
            (folder/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows if row['updates'] <= item['checkpoint_updates']))
            write(folder/'migration_owner.json', {'PID': os.getpid(), 'GPU': 0, 'deadline_unix': deadline,
                'checkpoint_updates': item['checkpoint_updates'], 'remote_stop': item,
                'remote_actual_wait': 'unknown because runner killed; not rewritten as success',
                'unsaved_tail': 'Preserved in remote_3090_snapshot; continuation starts at saved checkpoint'})
        argv = [PYTHON, '-B', '-u', '-m', 'runs.glisr_supplement_queue', 'resume-worker', '--method', method,
            '--seed', '2', '--deadline', str(deadline)]
        log = (folder/'GPU0_resume_stdout.log').open('w')
        child = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8'), stdout=log, stderr=subprocess.STDOUT)
        receipt = {'job': jid, 'PID': child.pid, 'GPU': 0, 'deadline_unix': deadline, 'argv': argv, 'phase': 'CONTINUE_SAVED_REMOTE_CHECKPOINT'}
        write(folder/'GPU0_resume_launch.json', receipt)
        active[jid] = {'child': child, 'log': log, 'deadline': deadline, 'method': method, 'phase': 'train'}
        write(attempt/'active.json', {j: {'PID': v['child'].pid, 'phase': v['phase'], 'deadline_unix': v['deadline']} for j, v in active.items()})
        while child.poll() is None and not (folder/'counted_resume_prefix50.json').exists():
            time.sleep(3)
    while active:
        for jid, record in list(active.items()):
            child, folder = record['child'], RUN/jid
            if time.time() > record['deadline'] and child.poll() is None:
                child.kill()
            if child.poll() is None:
                continue
            code = child.wait()
            write(folder/('GPU0_'+record['phase']+'_actual_wait.json'), {'PID': child.pid, 'actual_wait_exit': code,
                'hostname': os.uname().nodename, 'GPU': 0, 'continuation_from': str(folder/'remote_3090_snapshot/last.pt')})
            if code == 0 and record['phase'] == 'train':
                argv = [PYTHON, '-B', '-u', '-m', 'runs.glisr_cfc26_train', 'score', '--method', record['method'], '--seed', '2', '--deadline', str(record['deadline'])]
                record['child'] = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4'), stdout=record['log'], stderr=subprocess.STDOUT)
                record['phase'] = 'score'
                continue
            if code == 0:
                completion = json.loads((folder/'completion.json').read_text())
                write(folder/'results.json', {'status': 'COMPLETE', 'native_completion': completion, 'continuation': '3090savedstate toserverGPU0; unsavedtail rollback recorded'})
            write(folder/'remote_actual_wait.json', {'actual_wait_exit': code, 'hostname': os.uname().nodename,
                'GPU': 0, 'PID': child.pid, 'role': 'FINAL_CONTINUED_JOB_EXIT_ON_SERVER',
                'original_3090_actual_wait': 'unknown; see remote_3090_stop_receipt.json',
                'train_actual_wait': str(folder/'GPU0_train_actual_wait.json'), 'phase': record['phase']})
            record['log'].close()
            del active[jid]
        write(attempt/'active.json', {j: {'PID': v['child'].pid, 'phase': v['phase'], 'deadline_unix': v['deadline']} for j, v in active.items()})
        time.sleep(3)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', nargs='?', choices=['pipeline', 'worker', 'repair-waiting', 'resume-worker', 'migrate-remote'], default='pipeline')
    parser.add_argument('--method')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--attempt', default='engineering_wait_fix_v1')
    args = parser.parse_args()
    if args.phase == 'worker':
        cfc_worker(args.method, args.seed, args.deadline)
    elif args.phase == 'resume-worker':
        resume_worker(args.method, args.seed, args.deadline)
    elif args.phase == 'migrate-remote':
        migrate_remote()
    elif args.phase == 'repair-waiting':
        repair_waiting(args.attempt)
    else:
        pipeline()
