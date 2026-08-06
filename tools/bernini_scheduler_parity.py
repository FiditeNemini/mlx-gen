import argparse
import json
import platform
from pathlib import Path

import mlx.core as mx
import numpy as np

from mflux.models.wan.scheduler import WanUniPCMultistepScheduler


class BerniniSchedulerParity:
    REFERENCE_TIMESTEPS = [
        999,
        994,
        989,
        983,
        978,
        972,
        965,
        959,
        952,
        944,
        937,
        929,
        920,
        911,
        902,
        892,
        882,
        870,
        859,
        846,
        833,
        818,
        803,
        786,
        768,
        749,
        728,
        706,
        681,
        654,
        624,
        591,
        555,
        514,
        468,
        416,
        356,
        288,
        208,
        113,
    ]
    REFERENCE_SIGMAS = [
        0.9997998476028442,
        0.9946947693824768,
        0.9893770217895508,
        0.9838330149650574,
        0.9780480265617371,
        0.9720060229301453,
        0.9656894207000732,
        0.9590790867805481,
        0.9521540403366089,
        0.9448912739753723,
        0.9372654557228088,
        0.9292486906051636,
        0.9208100438117981,
        0.9119154810905457,
        0.9025267958641052,
        0.8926018476486206,
        0.8820932507514954,
        0.8709479570388794,
        0.8591062426567078,
        0.8465008735656738,
        0.8330553770065308,
        0.8186829090118408,
        0.803284227848053,
        0.7867453694343567,
        0.7689347267150879,
        0.7496998310089111,
        0.7288626432418823,
        0.7062143683433533,
        0.6815081238746643,
        0.6544499397277832,
        0.6246873140335083,
        0.5917934775352478,
        0.5552467703819275,
        0.5144029855728149,
        0.4684569239616394,
        0.4163888096809387,
        0.35688766837120056,
        0.28823959827423096,
        0.2081597000360489,
        0.11353304982185364,
        0.0,
    ]
    REFERENCE_STEP_SUMS = [
        27.449918746948242,
        26.86358642578125,
        25.020977020263672,
        19.023977279663086,
    ]
    REFERENCE_STEP_FIRST = [
        [-0.0062534395, 0.093746565, 0.19374657],
        [-0.030683802, 0.0693162, 0.16931622],
        [-0.10745925, -0.0074592466, 0.09254077],
        [-0.3573342, -0.25733417, -0.15733416],
    ]

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description="Compare the Bernini MLX UniPC schedule to Diffusers 0.35.2.")
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        report = BerniniSchedulerParity._report()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)

    @staticmethod
    def _report() -> dict:
        grid_scheduler = BerniniSchedulerParity._scheduler()
        grid_scheduler.set_timesteps(40)
        actual_timesteps = np.asarray(grid_scheduler.timesteps, dtype=np.int64)
        actual_sigmas = np.asarray(grid_scheduler.sigmas, dtype=np.float32)
        reference_timesteps = np.asarray(BerniniSchedulerParity.REFERENCE_TIMESTEPS, dtype=np.int64)
        reference_sigmas = np.asarray(BerniniSchedulerParity.REFERENCE_SIGMAS, dtype=np.float32)

        step_scheduler = BerniniSchedulerParity._scheduler()
        step_scheduler.set_timesteps(4)
        sample = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10
        actual_step_sums = []
        actual_step_first = []
        for index, timestep in enumerate(np.asarray(step_scheduler.timesteps, dtype=np.int64).tolist()):
            model_output = mx.full(sample.shape, 0.1 * (index + 1), dtype=mx.float32)
            sample = step_scheduler.step(model_output, timestep, sample, return_dict=False)[0]
            mx.eval(sample)
            actual_step_sums.append(float(mx.sum(sample).item()))
            actual_step_first.append(np.asarray(sample.reshape(-1)[:3], dtype=np.float32).tolist())

        step_sum_delta = np.abs(
            np.asarray(actual_step_sums, dtype=np.float64)
            - np.asarray(BerniniSchedulerParity.REFERENCE_STEP_SUMS, dtype=np.float64)
        )
        step_first_delta = np.abs(
            np.asarray(actual_step_first, dtype=np.float64)
            - np.asarray(BerniniSchedulerParity.REFERENCE_STEP_FIRST, dtype=np.float64)
        )
        sigma_delta = np.abs(actual_sigmas.astype(np.float64) - reference_sigmas.astype(np.float64))
        thresholds = {
            "max_sigma_absolute_error": 1e-7,
            "max_step_absolute_error": 2e-5,
        }
        grid_exact = bool(np.array_equal(actual_timesteps, reference_timesteps))
        max_sigma_error = float(sigma_delta.max())
        max_step_error = float(max(step_sum_delta.max(), step_first_delta.max()))
        return {
            "schema_version": 1,
            "kind": "bernini_diffusers_0_35_2_unipc_parity",
            "reference": {
                "package": "diffusers==0.35.2",
                "official_source_revision": "2d2b4591ac053ec25c6371b01a5a6746679e5793",
                "flow_shift": 5.0,
                "solver_order": 2,
                "solver_type": "bh2",
            },
            "environment": {
                "mlx_version": getattr(mx, "__version__", "unknown"),
                "platform": platform.platform(),
            },
            "grid_40": {
                "timesteps_exact": grid_exact,
                "actual_timesteps": actual_timesteps.tolist(),
                "reference_timesteps": reference_timesteps.tolist(),
                "max_sigma_absolute_error": max_sigma_error,
                "actual_sigmas": actual_sigmas.tolist(),
                "reference_sigmas": reference_sigmas.tolist(),
            },
            "solver_replay_4_steps": {
                "actual_sums": actual_step_sums,
                "reference_sums": BerniniSchedulerParity.REFERENCE_STEP_SUMS,
                "actual_first_values": actual_step_first,
                "reference_first_values": BerniniSchedulerParity.REFERENCE_STEP_FIRST,
                "max_absolute_error": max_step_error,
            },
            "thresholds": thresholds,
            "passed": (
                grid_exact
                and max_sigma_error <= thresholds["max_sigma_absolute_error"]
                and max_step_error <= thresholds["max_step_absolute_error"]
            ),
        }

    @staticmethod
    def _scheduler() -> WanUniPCMultistepScheduler:
        return WanUniPCMultistepScheduler(
            flow_shift=5.0,
            flow_sigma_schedule="diffusers-0.35.2",
        )


if __name__ == "__main__":
    BerniniSchedulerParity.main()
