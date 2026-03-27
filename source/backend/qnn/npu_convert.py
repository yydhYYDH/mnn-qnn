#!/usr/bin/python
import sys
import json
import os
import subprocess
import json
import shlex
post_treat = {}
qnn_sdk = os.environ["QNN_SDK_ROOT"]
print(qnn_sdk)
with open(sys.argv[1]) as f:
    post_treat = json.load(f)
soc_id = int(sys.argv[2])
dsp_arch = sys.argv[3]
print('soc_id:', soc_id, "; dsp_arch:", dsp_arch)
qnn_bin_path = os.path.join(qnn_sdk, 'bin', 'x86_64-linux-clang')
qnnModelLibGenerator = os.path.join(qnn_bin_path, 'qnn-model-lib-generator')
qnnContextBinaryGenerator = os.path.join(qnn_bin_path, 'qnn-context-binary-generator')
merges = post_treat["merge"]
cache_dir = 'res'
if 'cache' in post_treat:
    cache_dir = post_treat['cache']
clean_tmp = False

def run_subprocess(cmd, retries=3):
    result = None
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            return result
        print(f"[Retry {attempt}/{retries}] command failed: {cmd}")
    return result

def run_subprocess_parallel(tasks, retries=3):
    # tasks: list of (task_id, cmd)
    pending = list(tasks)
    succeeded = {}
    failures = {}
    for attempt in range(1, retries + 1):
        if not pending:
            break
        print(f"[Parallel Attempt {attempt}/{retries}] running {len(pending)} task(s)")
        procs = []
        for task_id, cmd in pending:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            procs.append((task_id, cmd, proc))

        next_pending = []
        for task_id, cmd, proc in procs:
            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout)
            if stderr:
                print(stderr)
            if proc.returncode == 0:
                succeeded[task_id] = True
                if task_id in failures:
                    del failures[task_id]
            else:
                print(f"[Retry {attempt}/{retries}] command failed: {cmd}")
                failures[task_id] = (cmd, proc.returncode)
                next_pending.append((task_id, cmd))
        pending = next_pending
    return succeeded, failures

context_config = {
    "backend_extensions": {
        "shared_library_path": os.path.join(qnn_sdk, "lib","x86_64-linux-clang","libQnnHtpNetRunExtensions.so"),
        "config_file_path": "./htp_backend_extensions.json"
    }
}
htp_so = os.path.join(qnn_sdk, 'lib','x86_64-linux-clang','libQnnHtp.so')
with open('context_config.json', 'w') as f:
    f.write(json.dumps(context_config, indent=4))

htp_backend_extensions = {
    "graphs": [
        {
            "vtcm_mb": 8,
            "O": 3.0,
            "fp16_relaxed_precision": 1,
            "hvx_threads": 4
        }
    ],
    "devices": [
        {
            "soc_id": soc_id,
            "dsp_arch": dsp_arch,
            "cores": [
                {
                    "core_id": 0,
                    "perf_profile": "burst",
                    "rpc_control_latency": 100
                }
            ]
        }
    ],
    "context": {
        "weight_sharing_enabled": True
    }
}

for key in post_treat["merge"]:
    srcs = merges[key]
    dst = key
    dstname = key.split('/')
    dstname = dstname[len(dstname)-1]
    dstname = dstname.replace('.bin', '')
    graphs = []
    libs = [None] * len(srcs)
    workdirs = []
    tar_tasks = []
    compile_tasks = []
    rm_raw_tasks = []
    rm_bin_tasks = []
    for i,src in enumerate(srcs):
        graphname = src.split('/')
        graphname = graphname[len(graphname)-1]
        graphs.append(graphname)
        workdir = os.path.join(os.getcwd(), src)
        workdirs.append(workdir)
        q_workdir = shlex.quote(workdir)
        q_graph = shlex.quote(graphname)
        tar_cmd = "cd " + q_workdir + " && tar -cf " + q_graph + ".bin *.raw"
        tar_tasks.append((i, tar_cmd))
        if clean_tmp:
            rm_raw_cmd = "cd " + q_workdir + " && rm *.raw"
            rm_raw_tasks.append((i, rm_raw_cmd))
        compile_cmd = 'python3 ' + qnnModelLibGenerator + ' -c ' + os.path.join(workdir, graphname + '.cpp') + ' -b ' + os.path.join(workdir, graphname + '.bin') + ' -t x86_64-linux-clang -o ' + workdir
        compile_tasks.append((i, compile_cmd))
        if clean_tmp:
            rm_bin_cmd = "rm " + shlex.quote(os.path.join(workdir, graphname + ".bin"))
            rm_bin_tasks.append((i, rm_bin_cmd))
        libs[i] = os.path.join(workdir, 'x86_64-linux-clang', 'lib' + graphname + '.so')

    _, tar_failures = run_subprocess_parallel(tar_tasks, retries=1)
    if tar_failures:
        first_id = next(iter(tar_failures))
        cmd, code = tar_failures[first_id]
        raise RuntimeError(f"Tar failed for graph index {first_id}, exit_code={code}, cmd={cmd}")

    if clean_tmp and rm_raw_tasks:
        _, rm_raw_failures = run_subprocess_parallel(rm_raw_tasks, retries=1)
        if rm_raw_failures:
            first_id = next(iter(rm_raw_failures))
            cmd, code = rm_raw_failures[first_id]
            raise RuntimeError(f"Remove raw failed for graph index {first_id}, exit_code={code}, cmd={cmd}")

    _, compile_failures = run_subprocess_parallel(compile_tasks, retries=3)
    if compile_failures:
        first_id = next(iter(compile_failures))
        cmd, code = compile_failures[first_id]
        raise RuntimeError(f"Compile failed for graph index {first_id}, exit_code={code}, cmd={cmd}")

    if clean_tmp and rm_bin_tasks:
        _, rm_bin_failures = run_subprocess_parallel(rm_bin_tasks, retries=1)
        if rm_bin_failures:
            first_id = next(iter(rm_bin_failures))
            cmd, code = rm_bin_failures[first_id]
            raise RuntimeError(f"Remove bin failed for graph index {first_id}, exit_code={code}, cmd={cmd}")

    if any(lib is None for lib in libs):
        raise RuntimeError("Some graph libraries were not generated successfully.")
    htp_backend_extensions['graphs'][0]['graph_names'] = graphs
    with open('htp_backend_extensions.json', 'w') as f:
        f.write(json.dumps(htp_backend_extensions, indent=4))
    libsStr = ""
    for i in range(0, len(libs)):
        if i > 0:
            libsStr+=','
        libsStr += libs[i]
    context_cmd = qnnContextBinaryGenerator + ' --model ' + libsStr + ' --backend ' + htp_so + ' --binary_file ' + dstname + ' --config_file ./context_config.json ' + ' --output_dir ' + cache_dir
    context_result = run_subprocess(context_cmd, retries=3)
    if context_result.returncode != 0:
        raise RuntimeError(f"Context binary generation failed: {context_cmd}")
    if clean_tmp:
        for workdir in workdirs:
            rm_workdir_cmd = "rm -rf " + workdir
            rm_workdir_result = run_subprocess(rm_workdir_cmd, retries=1)
            if rm_workdir_result.returncode != 0:
                raise RuntimeError(f"Remove workdir failed: {rm_workdir_cmd}")
