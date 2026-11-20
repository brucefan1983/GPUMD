#!/usr/bin/env python3
# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD. If not, see <https://www.gnu.org/licenses/>.
# Major contributors: Zheyong Fan (NEP multi-GPU release tests).


"""Re-evaluate retained outputs without launching GPUMD or modifying old reports."""

import argparse
import json
from pathlib import Path
import sys

from checks import CheckError, check_liquid, exact, load_tolerances
from run_tests import assess, performance_summary, save_report, stamp, verify_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='original result directory')
    parser.add_argument('--tolerances', type=Path, help='explicit replacement bounds; default: original bounds')
    args = parser.parse_args()
    root = args.results.resolve()
    report = json.loads((root / 'report.json').read_text())
    exact(report['schema_version'] == 1, 'Unsupported report version')
    tolerances = load_tolerances(args.tolerances) if args.tolerances else report['tolerances']
    report['rechecked_utc'] = stamp()
    report['original_results'] = str(root)
    # Re-evaluate the liquid gate too; preparation failure must never silently
    # become a passing liquid test after a numerical tolerance change.
    for run in report['runs']:
        if run['spec']['kind'] == 'preparation':
            try:
                verify_record(root, run)
                report['liquid'] = check_liquid(root / run['path'])
            except (OSError, ValueError, CheckError) as error:
                report['fatal_error'] = 'Liquid preparation: ' + str(error)
    assess(root, report, tolerances)
    performance_summary(root, report)
    destination = root / ('recheck-' + stamp())
    save_report(destination, report)
    for check in report['correctness_checks']:
        print('{} {}'.format(check['status'], check['name']))
    print('GPUMD was not launched. Results: ' + str(destination / 'summary.md'))
    return int(bool(report.get('fatal_error')) or report['correctness_status'] in ('FAIL', 'ERROR') or
               report['performance_status'] == 'ERROR')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, CheckError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
