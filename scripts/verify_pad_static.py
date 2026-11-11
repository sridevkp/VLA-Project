"""Quick static pad-position measurement (no arm motion beyond reset)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from vlarobot.backends.webots.config import CAN_DEF
from vlarobot.backends.webots.sim_kinematics import PROTO_PAD_BOTTOM_ARM7_M, PROTO_PAD_CENTER_ARM7_M
from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.episode_runner import (
    _popen,_stop_client,_terminate_process,_write_pid_file,build_webots_command,resolve_webots,wait_for_simulator,webots_environment,
)

def _reserve_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--webots',required=True); ap.add_argument('--world',default=str(Path(__file__).resolve().parents[1]/'worlds/factory.wbt')); ap.add_argument('--out',default='runs/campaign-2026-11-11/phase2/pad_static.json'); ap.add_argument('--startup-timeout',type=float,default=180.0); args=ap.parse_args()
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    webots=resolve_webots(args.webots); world=Path(args.world).resolve(); port=_reserve_port(); endpoint=f'tcp://127.0.0.1:{port}'
    log_path=out.with_suffix('.log'); pid_path=out.with_suffix('.pid')
    with log_path.open('w',encoding='utf-8') as logf:
        env=webots_environment(webots); env['VLAROBOT_ZMQ_PORT']=str(port)
        proc=_popen(build_webots_command(webots,world,headless=False), env, logf); _write_pid_file(pid_path, proc.pid)
        client=None
        try:
            client=wait_for_simulator(endpoint,proc,args.startup_timeout)
            obs=client.reset(ScenarioSpec(), seed=0)
            for _ in range(40): obs=client.step(1)
            gt=client.get_ground_truth(); can=gt.object_poses[CAN_DEF]
            can_world=[can.position.x,can.position.y,can.position.z]
            d=obs.robot.diagnostics
            out.write_text(json.dumps({'status':'ok','sim_time_s':obs.t,'can_world_xyz':can_world,'diagnostics':d,'proto_pad_center_arm7_m':PROTO_PAD_CENTER_ARM7_M.tolist(),'proto_pad_bottom_arm7_m':PROTO_PAD_BOTTOM_ARM7_M.tolist()}, default=str, indent=2), encoding='utf-8')
            print(json.dumps({'out':str(out),'has_finger_midpoint':'finger_midpoint_world_xyz_m' in d,'has_kinematics_telemetry':'kinematics_telemetry' in d}))
            return 0
        finally:
            if client is not None: _stop_client(client)
            _terminate_process(proc,30.0,pid_path=pid_path)

if __name__=='__main__': raise SystemExit(main())
