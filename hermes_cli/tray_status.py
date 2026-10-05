"""Structured tray status, including official upstream comparison for local branches."""
import contextlib
import io
import json
from pathlib import Path
import sys


def read_status(root, *, force=False):
    from hermes_cli._startup_fast import print_fast_version_info
    from hermes_cli.source_check import check_for_updates

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_fast_version_info(check_updates=False)
        status = check_for_updates(install_root=root, force=force)
        local_branch = None
        if status.get('error') == 'branch-local-only':
            local_branch = status.get('currentBranch') or status.get('branch')
            # An explicit passive target never changes the checkout or heals a branch.
            # Keep canonical remote-tip resolution and the carried merge-base counting.
            status = check_for_updates(install_root=root, branch='main', force=force)
    return {'versionText': out.getvalue(), 'supported': status.get('supported', False),
            'updateAvailable': status.get('updateAvailable'), 'behind': status.get('behind'),
            'error': status.get('error'), 'reason': status.get('reason'),
            'localBranch': local_branch, 'comparisonBranch': status.get('branch'),
            'currentSha': status.get('currentSha'), 'targetSha': status.get('targetSha'),
            'commits': status.get('commits') or []}


def main():
    root = Path(sys.argv[1]).resolve()
    print(json.dumps(read_status(root, force='--force' in sys.argv)))


if __name__ == '__main__':
    main()
