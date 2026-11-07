"""One fixed source-supervised reference and two development readout families."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torchvision

from sonar_encoder.training.model import build_encoder, prepare_images
from sonar_encoder.training.readout import state_digest

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'shared_supervised_reference_v1'
AUTH = ROOT / '.aris/traces/experiment-plan/2026-10-07_shared_supervised_reference/contract.json'
DESIGN = BASE / 'shared_supervised_reference_contract_v1/contract.json'
INITIAL = 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e'
WEIGHTS = ROOT / 'assets/dinov2_vits14_pretrain.pth'
CODE = ROOT / 'assets/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
REAL = BASE.parent / 'incoming/real_fls'
FSOD = REAL / 'task_validation/fsod_task_ready'
FSOD_IMAGES = REAL / 'baidu_downloads/FSOD_complete_20260928'
ARMS = ('E_U', 'E_M', 'E_UM')
STATES = ('T0', *ARMS)
TASKS = ('fsod', 'nksid')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_protocol():
    return json.loads((RUN / 'protocol.json').read_text())


def configure_runtime(task):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(task == 'nksid')
    torch.set_float32_matmul_precision('highest')


def resources():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = torch.cuda.max_memory_reserved(0) if torch.cuda.is_initialized() else 0
    disk = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    assert rss <= 32 * 1024**3 and gpu <= 16 * 1024**3 and disk <= 16 * 1024**3, (rss, gpu, disk)
    return {'RSS_peak_bytes': rss, 'GPU_peak_reserved_bytes': gpu, 'disk_bytes': disk}


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    resources()
    temp.replace(path)


def atomic_predictions(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as output:
        np.save(output, value, allow_pickle=False)
    resources()
    temp.replace(path)


def load_initial_encoder():
    torch.manual_seed(0)
    model = build_encoder(CODE, WEIGHTS, 'R010')
    assert state_digest(model.state_dict()) == INITIAL
    return model


def load_frozen_encoder(state):
    model = load_initial_encoder()
    if state != 'T0':
        path = RUN / 'arms' / state / 'last.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == digest(RUN / 'protocol.json')
        assert saved['clock']['head_success'] == 200 and saved['clock']['body_success'] == 6000
        report = json.loads((path.parent / 'report_body6000.json').read_text())
        model.load_state_dict(saved['encoder'], strict=True)
        assert state_digest(model.state_dict()) == saved['encoder_state_sha256'] == report['encoder_state_sha256']
    return model.eval().requires_grad_(False)


def gray_bytes(path, expected, shape):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == expected, path
    with Image.open(io.BytesIO(raw)) as image:
        gray = np.array(image.convert('L'), dtype=np.uint8)
    assert gray.shape == tuple(shape), (path, gray.shape, shape)
    return gray


def prepare():
    configure_runtime('source')
    from runs.shared_supervised_train import init_source_heads
    from runs.shared_supervised_readout import make_head

    assert digest(AUTH) == '9b57bd4b2e636a5b034b11db4974e990dcfaa842ae8fd080e88b79c66e526890'
    assert digest(DESIGN) == '03b03905686b7f5568aa9eb3dd897e70446a4dba26f17d09e0bfdf7ec9200ec0'
    auth = json.loads(AUTH.read_text())
    design = json.loads(DESIGN.read_text())
    source_inputs, targets, bindings = {}, {}, {str(AUTH): digest(AUTH), str(DESIGN): digest(DESIGN)}
    data_sources = {'U': ROOT/'runs/data/uatd_readout_v1', 'M': ROOT/'runs/data/readout_v1/marine',
                    'nksid': ROOT/'runs/data/readout_v1/nksid'}
    expected_manifests = {'U': design['source_data']['U']['expected_manifest_sha256'],
                          'M': design['source_data']['M']['expected_manifest_sha256'],
                          'nksid': design['target_readouts']['NKSID']['expected_manifest_sha256']}
    for name, data in data_sources.items():
        path, norm_path = data/'manifest.json', data/'normalization.json'
        bindings[str(path)], bindings[str(norm_path)] = digest(path), digest(norm_path)
        assert bindings[str(path)] == expected_manifests[name]
        manifest, norm = json.loads(path.read_text()), json.loads(norm_path.read_text())
        assert norm['source_manifest_sha256'] == bindings[str(path)]
        expected_norm = design['source_data'][name] if name != 'nksid' else design['target_readouts']['NKSID']
        assert (norm['mean'], norm['std']) == (expected_norm['mean'], expected_norm['std'])
        if name == 'nksid':
            rows = {split: [{**row, 'image_sha256': row['sha256']} for row in manifest[split]]
                    for split in ('train', 'validation')}
            assert [len(rows[s]) for s in ('train', 'validation')] == [1680, 417]
            assert manifest['class_names'] == design['target_readouts']['NKSID']['class_names']
            for split in rows:
                assert [r['row_id'] for r in rows[split]] == sorted(r['row_id'] for r in rows[split])
                for row in rows[split]:
                    gray_bytes(row['image'], row['image_sha256'], row['shape'][:2])
            targets['nksid'] = {'rows': rows, 'normalization': norm, 'class_names': manifest['class_names']}
            continue
        rows = manifest['train']
        assert len(rows) == (6051 if name == 'U' else 1000)
        if name == 'U':
            rows = [{k: row[k] for k in ('row_id', 'image', 'shape', 'labels', 'boxes', 'image_sha256')} for row in rows]
            for row in rows:
                boxes = np.array(row['boxes'], dtype=np.float64).reshape(-1, 4)
                assert len(boxes) > 0 and len(boxes) == len(row['labels'])
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all()
                assert ((boxes >= 0) & (boxes <= 1)).all() and all(0 <= label < 10 for label in row['labels'])
        else:
            rows = [{**row, 'image_sha256': row['imageSHA256'], 'mask_sha256': row['maskSHA256']} for row in rows]
            for row in rows:
                assert row['shape'] == [480, 320]
                raw = Path(row['mask']).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == row['mask_sha256']
                with Image.open(io.BytesIO(raw)) as image:
                    mask = np.array(image)
                assert mask.dtype == np.uint8 and mask.shape == (480, 320) and mask.max() <= 11
        path = RUN/'inputs'/f'{name}.npy'
        array = np.lib.format.open_memmap(path.with_suffix('.npy.partial'), mode='w+', dtype=np.float32,
                                        shape=(len(rows), 1, 224, 224))
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset+8]
            samples = []
            for row in batch:
                gray = gray_bytes(row['image'], row['image_sha256'], row['shape'][:2])
                values = torch.from_numpy(gray).float().unsqueeze(0) / 255
                samples.append({'intensity': values, 'valid': torch.ones(gray.shape, dtype=torch.bool)})
            images = prepare_images(samples, norm, None)
            assert images.dtype == torch.float32 and torch.isfinite(images).all()
            array[offset:offset+len(batch)] = images.numpy()
            if offset % 400 == 0:
                resources()
                print(json.dumps({'phase': 'prepare', 'source': name, 'rows': offset+len(batch)}), flush=True)
        array.flush()
        del array
        path.with_suffix('.npy.partial').replace(path)
        source_inputs[name] = {'rows': rows, 'normalization': norm, 'view_path': str(path), 'view_sha256': digest(path),
                               'class_names': manifest['class_names']}
    fsod_rows = {}
    total, squared, count = 0., 0., 0
    for split, expected_count in (('train', 2357), ('validation', 785)):
        source_split = 'val' if split == 'validation' else 'train'
        path = FSOD/f'instances_{source_split}_similarity_control.json'
        bindings[str(path)] = digest(path)
        coco = json.loads(path.read_text())
        assert [c['id'] for c in coco['categories']] == list(range(1, 11))
        annotations = defaultdict(list)
        for annotation in coco['annotations']:
            assert annotation['category_id'] in range(1, 11) and annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0
            annotations[annotation['image_id']].append(annotation)
        rows = []
        for image in sorted(coco['images'], key=lambda r: r['id']):
            assert (image['width'], image['height']) == (2008, 1112)
            image_path = FSOD_IMAGES/source_split/image['file_name']
            raw = image_path.read_bytes()
            with Image.open(io.BytesIO(raw)) as source:
                gray = np.array(source.convert('L'), dtype=np.uint8)
            assert gray.shape == (1112, 2008)
            if split == 'train':
                values = np.log1p(gray.astype(np.float64) / 255)
                total += float(values.sum()); squared += float(np.square(values).sum()); count += values.size
            rows.append({'image_id': image['id'], 'image': str(image_path), 'width': 2008, 'height': 1112,
                         'image_sha256': hashlib.sha256(raw).hexdigest(), 'annotations': annotations[image['id']]})
        assert len(rows) == expected_count
        fsod_rows[split] = rows
        print(json.dumps({'phase': 'prepare', 'target': 'fsod', 'split': split, 'rows': len(rows)}), flush=True)
    mean = total/count
    norm = {'mean': mean, 'std': float(np.sqrt(squared/count-mean*mean)), 'pixel_count': count,
            'domain': 'FP64 native2357train-only log1p(PIL L uint8/255) population; no validation pixels.'}
    assert count == 2357*2008*1112 and norm['std'] > 0
    write_json(RUN/'FSOD_normalization.json', norm)
    targets['fsod'] = {'rows': fsod_rows, 'normalization': norm,
                       'training_coco': str(FSOD/'instances_train_similarity_control.json'),
                       'validation_coco': str(FSOD/'instances_val_similarity_control.json')}
    encoder = load_initial_encoder()
    initial = state_digest(encoder.state_dict())
    actual_config = {'variant': 'R010', 'embed_dim': encoder.embed_dim,
                     'body_parameter_count': sum(p.numel() for p in encoder.parameters()),
                     'register_tokens': encoder.backbone.num_register_tokens,
                     'block_count': len(encoder.backbone.blocks),
                     'block_module_types': [type(block).__name__ for block in encoder.backbone.blocks],
                     'drop_path_module_types': [type(block.drop_path1).__name__ for block in encoder.backbone.blocks],
                     'position_embedding_shape': list(encoder.backbone.pos_embed.shape),
                     'final_norm_epsilon': encoder.backbone.norm.eps, 'source_norms_separate': True}
    assert actual_config['block_count'] == 12 and set(actual_config['drop_path_module_types']) == {'Identity'}
    del encoder
    source_heads = init_source_heads('E_UM')
    source_head_hashes = {name: state_digest(head.state_dict()) for name, head in source_heads.items()}
    del source_heads
    for task in TASKS:
        torch.manual_seed(0)
        head = make_head(task)
        targets[task]['initial_head_sha256'] = state_digest(head.state_dict())
        targets[task]['actual_head_parameter_count'] = sum(p.numel() for p in head.parameters())
        del head
    sources = [Path(__file__), ROOT/'runs/shared_supervised_train.py', ROOT/'runs/shared_supervised_readout.py',
               ROOT/'tests/test_shared_supervised_train.py', ROOT/'tests/test_shared_supervised_readout.py',
               ROOT/'sonar_encoder/training/model.py', ROOT/'sonar_encoder/training/uatd_head.py',
               ROOT/'sonar_encoder/training/readout.py', ROOT/'runs/cfc_frozen_detector.py',
               ROOT/'runs/cfc_independent_ap50.py',
               RUN/'preflight/focused_CPU.log', RUN/'preflight/focused_CPU.time', RUN/'preflight/source_review.json']
    bindings.update({str(path): digest(path) for path in sources})
    bindings.update({str(path): digest(path) for path in sorted(CODE.rglob('*.py'))})
    operator_library = Path(torchvision.extension._get_extension_path('_C'))
    assert torchvision.extension._HAS_OPS
    bindings[str(operator_library)] = digest(operator_library)
    write_json(RUN/'runtime_binding.json', {'executable': sys.executable, 'python': sys.version,
               'torch': str(torch.__version__), 'CUDA_version': torch.version.cuda, 'torch_file': torch.__file__,
               'numpy': np.__version__, 'torchvision_version': torchvision.__version__,
               'torchvision_file': torchvision.__file__, 'torchvision_operator_library': str(operator_library),
               'torchvision_operator_library_sha256': bindings[str(operator_library)],
               'torch_loaded_libraries': sorted(torch.ops.loaded_libraries),
               'CUDA_initialized_during_prepare': torch.cuda.is_initialized(),
               'workspace': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'GPU_affinity': os.environ['CUDA_VISIBLE_DEVICES']})
    assert not torch.cuda.is_initialized()
    protocol = {'unit': 'shared_supervised_reference_v1', 'authorization_sha256': digest(AUTH), 'design_sha256': digest(DESIGN),
                'contract': auth, 'source_inputs': source_inputs, 'targets': targets, 'inputs_sha256': bindings,
                'initial_encoder_sha256': initial, 'actual_initial_encoder_config': actual_config,
                'initial_source_head_sha256': source_head_hashes, 'encoder_weights_path': str(WEIGHTS),
                'encoder_weights_sha256': digest(WEIGHTS), 'code_dir': str(CODE),
                'source_recipe': design['training_policy'], 'source_losses': design['exact_source_losses'],
                'sampling': design['sampling'], 'readout_primary': auth['adopt_parent_except']['readout_primary'],
                'prediction_evidence': auth['adopt_parent_except']['prediction_evidence'],
                'runtime_binding_sha256': digest(RUN/'runtime_binding.json'), 'freeze_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(RUN/'protocol.json', protocol)
    (RUN/'source').mkdir()
    for path in sources:
        if path.suffix == '.py': shutil.copyfile(path, RUN/'source'/path.name)
    write_json(RUN/'prepare_receipt.json', {'status': 'FROZEN_BEFORE_FIRST_FORMAL_FORWARD',
               'protocol_sha256': digest(RUN/'protocol.json'), 'initial_encoder_sha256': initial,
               'source_population': {'U': 6051, 'M': 1000}, 'target_population': {'fsod': [2357,785], 'nksid': [1680,417]},
               'source_view_bytes': 7051*224*224*4, 'FSOD_normalization': norm,
               'head_initial_sha256': {'source': source_head_hashes, 'target': {t: targets[t]['initial_head_sha256'] for t in TASKS}},
               'resources': resources()})
    print(json.dumps({'status': 'FROZEN', 'protocol_sha256': digest(RUN/'protocol.json')}), flush=True)


def ledger():
    return [json.loads(x) for x in (RUN/'compute_ledger.jsonl').read_text().splitlines()]


def remaining_seconds(group):
    records = ledger()
    total = 129600-sum(x['seconds'] for x in records)
    used = lambda prefix: sum(x['seconds'] for x in records if x['group'].startswith(prefix))
    exact = sum(x['seconds'] for x in records if x['group'] == group)
    if group.startswith('adapt:'): return min(total, 21600-exact, 64800-used('adapt:'))
    if group.startswith('fsod:'): return min(total, 7200-exact, 28800-used('fsod:'))
    if group.startswith('nksid:'): return min(total, 300-exact, 1200-used('nksid:'))
    return min(total, {'prepare': 5400, 'encode': 14400, 'score': 7200}[group]-exact)


def gpu_entry(phase):
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    gpu0 = next(line.split(', ') for line in query.splitlines() if line.startswith('0, '))
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    conflicts, foreign, disappeared = [], [], []
    for line in processes.splitlines():
        uuid, pid, executable, memory = line.split(', ')
        if uuid == gpu0[1] and int(pid) != os.getpid():
            proc = Path('/proc')/pid
            try:
                owner = proc.stat().st_uid
                if owner != os.getuid():
                    foreign.append({'pid': int(pid), 'UID': owner})
                elif str((proc/'cwd').resolve(strict=True)) == str(ROOT):
                    conflicts.append(int(pid))
            except FileNotFoundError:
                disappeared.append(int(pid))
    result = {'phase': phase, 'GPU_query': query, 'process_metadata': processes, 'foreign_UIDs': foreign,
              'same_project_conflicts': conflicts, 'disappeared_PIDs': disappeared,
              'free_bytes': int(gpu0[2])*1024**2}
    result['passed'] = result['free_bytes'] >= 24*1024**3 and not conflicts
    with (RUN/'gpu_entries.jsonl').open('a') as output: output.write(json.dumps(result)+'\n')
    return result['passed']


def execute(phase, group, state=None, task=None, stop=None):
    if phase in ('train','extract','fit'):
        admission_started = time.monotonic()
        admission = {'phase': 'GPU_entry', 'group': group, 'state': state, 'task': task,
                     'passed': None}
        try:
            admission['passed'] = gpu_entry(f'{phase}:{state}:{task}')
        finally:
            admission['seconds'] = time.monotonic()-admission_started
            with (RUN/'compute_ledger.jsonl').open('a') as output: output.write(json.dumps(admission)+'\n')
        if not admission['passed']:
            write_json(RUN/'deferred.json', {'status': 'DEFERRED_AT_GPU_BOUNDARY_NO_QUEUE', 'phase': phase,
                       'state': state, 'task': task, 'admission': admission})
            return False
    allowed = remaining_seconds(group)
    assert allowed > 0
    command = [sys.executable, '-m', 'runs.shared_supervised_reference', phase]
    if state is not None: command += ['--state',state]
    if task is not None: command += ['--task',task]
    if stop is not None: command += ['--stop',str(stop)]
    name = '_'.join(str(x) for x in (phase,state,task,stop) if x is not None)
    log = RUN/(name+'.log')
    started = time.monotonic()
    with log.open('w') as output:
        process = subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'phase': phase, 'state': state, 'task': task, 'pid': process.pid, 'log': str(log), 'seconds_available': allowed}), flush=True)
        rss = 0
        while process.poll() is None:
            if time.monotonic()-started > allowed:
                os.killpg(process.pid, signal.SIGTERM); process.wait(); break
            status = Path(f'/proc/{process.pid}/status')
            if status.exists():
                fields = {l.split(':')[0]:l.split(':',1)[1].strip() for l in status.read_text().splitlines() if ':' in l}
                if 'VmHWM' in fields:
                    rss = max(rss, int(fields['VmHWM'].split()[0])*1024)
                    if rss + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 32*1024**3:
                        os.killpg(process.pid, signal.SIGTERM); process.wait(); break
            time.sleep(2)
        code = process.wait()
    row = {'phase': phase, 'group': group, 'state': state, 'task': task, 'seconds': time.monotonic()-started,
           'pid': process.pid, 'exit_code': code, 'RSS_peak_observed_bytes': rss, 'closed_log_sha256': digest(log)}
    with (RUN/'compute_ledger.jsonl').open('a') as output: output.write(json.dumps(row)+'\n')
    if code != 0:
        write_json(RUN/'failure_closure.json', {'status': 'FAILED_PHASE_STOP_NO_RETRY', 'phase': row, 'ledger': ledger()})
        return False
    assert row['seconds'] <= allowed and sum(x['seconds'] for x in ledger()) <= 129600
    return True


def gate():
    reports = {arm: json.loads((RUN/'arms'/arm/'report_body50.json').read_text()) for arm in ARMS}
    for arm, report in reports.items():
        assert report['clock']['head_success'] == 200 and report['clock']['body_success'] == 50
        assert report['encoder_state_sha256'] != INITIAL
        assert report['encoder_training'] is True and report['body_requires_grad_except_mask'] is True
        assert report['mask_token_in_optimizer'] is False and report['mask_token_requires_grad'] is False
        assert report['initial_mask_token_sha256'] == report['mask_token_sha256']
        for value in report['body_gradient_witness'].values():
            assert value['requires_grad'] and value['in_optimizer'] and value['gradient_finite']
        assert any(value['gradient_l2_norm'] > 0 and value['weight_movement_abs_max'] > 0
                   for value in report['body_gradient_witness'].values())
    rows = ledger()
    projections = {}
    for arm in ARMS:
        head = json.loads((RUN/'arms'/arm/'report_head200.json').read_text())
        # Use actual body work, rather than averaging it with cheaper frozen-body head initialization.
        body_elapsed = reports[arm]['clock']['elapsed_seconds'] - head['clock']['elapsed_seconds']
        projected = 1.5 * body_elapsed / 50 * 5950
        projections[arm] = projected
        assert projected <= remaining_seconds('adapt:'+arm)
    assert sum(projections.values()) <= 64800-sum(r['seconds'] for r in rows if r['group'].startswith('adapt:'))
    assert sum(projections.values()) + 14400 + 28800 + 1200 + 7200 <= 129600-sum(r['seconds'] for r in rows)
    write_json(RUN/'engineering_gate.json', {'status': 'THREE_SOURCE_PREFIXES_PASS_NO_QUALITY_SELECTION', 'passed': True,
               'reports_sha256': {arm: digest(RUN/'arms'/arm/'report_body50.json') for arm in ARMS},
               'projected_remaining_source_seconds': projections, 'projection_is_conditional_not_measured_completion': True,
               'resources': resources()})


def close():
    from runs.shared_supervised_readout import score_and_summarize
    score_and_summarize()
    protocol = load_protocol()
    reports = {arm: json.loads((RUN/'arms'/arm/'report_body6000.json').read_text()) for arm in ARMS}
    for arm, report in reports.items():
        assert report['clock']['head_success'] == 200 and report['clock']['body_success'] == 6000
        for stage, count in (('HEAD_INIT',6400),('BODY_FT',192000)):
            completed = sum(x['completed']['visits'] for x in report['source_exposure'][stage].values())
            effective = sum(x['effective']['visits'] for x in report['source_exposure'][stage].values())
            assert completed == effective == count
    results = json.loads((RUN/'target_results.json').read_text())
    assert results['logical_endpoints'] == 16
    write_json(RUN/'results.json', {'status': 'COMPLETE_FIXED_SUPERVISED_REFERENCE', 'source_reports': reports,
               'targets': results, 'terminal_primary_best_auxiliary': True, 'protocol_sha256': digest(RUN/'protocol.json'),
               'not_novel_or_strongest_or_convergence_acceptance': True})
    write_json(RUN/'verification.json', {'status': 'PASS_SIXTEEN_LOGICAL_SAVED_PREDICTION_ENDPOINTS',
               'results_sha256': digest(RUN/'results.json'), 'target_results_sha256': digest(RUN/'target_results.json'),
               'inherited_inputs_initial_bindings_not_whole_corpus_rehash': protocol['inputs_sha256']})
    write_json(RUN/'completion.json', {'status': 'COMPLETE_STOP_NO_NEW_PHASE', 'results_sha256': digest(RUN/'results.json'),
               'verification_sha256': digest(RUN/'verification.json'), 'protocol_sha256': digest(RUN/'protocol.json'),
               'ledger_seconds_before_closing_process_exit': sum(r['seconds'] for r in ledger()), 'resources': resources()})


def pipeline():
    if not execute('prepare','prepare'): return
    for arm in ARMS:
        if not execute('train','adapt:'+arm,arm,stop=50): return
    if not execute('gate','prepare'): return
    for arm in ARMS:
        if not execute('train','adapt:'+arm,arm,stop=6000): return
    for state in STATES:
        for task in TASKS:
            if not execute('extract','encode',state,task): return
            if not execute('fit',task+':'+state,state,task): return
            if not execute('score','score',state,task): return
    if not execute('close','score'): return
    closed = json.loads((RUN/'completion.json').read_text())
    closed['ledger_final_seconds'] = sum(row['seconds'] for row in ledger())
    closed['formal_phase_exits'] = [{k: row[k] for k in ('phase','state','task','pid','exit_code')} for row in ledger() if 'pid' in row]
    write_json(RUN/'completion.json', closed)
    print(json.dumps({'status': closed['status'], 'completion_sha256': digest(RUN/'completion.json')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','train','gate','extract','fit','score','close','pipeline'))
    parser.add_argument('--state')
    parser.add_argument('--task')
    parser.add_argument('--stop',type=int)
    args = parser.parse_args()
    if args.phase == 'pipeline': pipeline()
    elif args.phase == 'prepare': prepare()
    elif args.phase == 'train':
        from runs.shared_supervised_train import train_arm
        train_arm(args.state,args.stop)
    elif args.phase == 'gate': gate()
    elif args.phase == 'close': close()
    else:
        from runs.shared_supervised_readout import targetextract,fit_head,score_readout
        {'extract':targetextract,'fit':fit_head,'score':score_readout}[args.phase](args.state,args.task)
