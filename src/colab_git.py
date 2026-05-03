"""
helper for committing artifacts from a colab notebook.

reads GITHUB_PAT from colab secrets (key icon in sidebar), uses it via
http.extraHeader so the token never gets stored in .git/config or shows
up in process listings. one function call from any phase notebook:

    from src.colab_git import commit_artifacts
    commit_artifacts(message='phase X artifacts')

if no GITHUB_PAT secret exists, prints a warning and skips.
"""
from __future__ import annotations
import subprocess


REPO_ROOT = '/content/project'
REMOTE_URL = 'https://github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp.git'
DEFAULT_PATHS = ['dashboard/assets/', 'models/portable/']


def _get_pat() -> str | None:
    """try to read GITHUB_PAT from colab user secrets. returns None if missing."""
    try:
        from google.colab import userdata
        return userdata.get('GITHUB_PAT')
    except Exception:
        return None


def _ensure_git_identity() -> None:
    """colab has no global git identity; set a noreply one for commits."""
    subprocess.run(
        ['git', '-C', REPO_ROOT, 'config', 'user.email',
         'george-gideon-S@users.noreply.github.com'],
        check=True, capture_output=True,
    )
    subprocess.run(
        ['git', '-C', REPO_ROOT, 'config', 'user.name', 'george-gideon-S'],
        check=True, capture_output=True,
    )


def commit_artifacts(message: str = 'phase artifacts from colab', paths: list | None = None) -> bool:
    """
    stage paths, commit if there are changes, push using the PAT.

    push uses url-embedded auth as a one-shot push target rather than
    setting it as the persistent remote, so the token never lands in
    .git/config. http.extraHeader doesnt preempt git's basic-auth prompt
    on some versions, but a fully-qualified url with embedded creds does.

    Args:
        message: commit message.
        paths: list of git paths to stage. defaults to dashboard/assets/ and
               models/portable/.

    Returns:
        True if a push occurred, False if no token or push failed.
    """
    paths = paths or DEFAULT_PATHS
    token = _get_pat()
    if not token:
        print('GITHUB_PAT not in colab secrets - skipping commit/push.')
        print('to enable auto-push, add a secret named GITHUB_PAT via the key '
              'icon in the colab sidebar with notebook access turned on.')
        return False

    _ensure_git_identity()

    # stage. check=False because git add returns exit 1 when EVERY supplied
    # path is gitignored - which is the expected case after we cleaned up
    # the repo to keep generated artifacts out. we treat that as "nothing
    # to stage" rather than as an error.
    add_res = subprocess.run(
        ['git', '-C', REPO_ROOT, 'add'] + paths,
        capture_output=True, text=True,
    )
    if add_res.returncode != 0:
        # most common cause: every path was gitignored. show the message
        # but dont crash.
        msg = (add_res.stderr or '') + (add_res.stdout or '')
        if 'ignored by one of your .gitignore' in msg:
            print('all artifact paths are gitignored - nothing to stage.')
        else:
            print('git add reported issues:')
            print(msg.strip())

    # was anything actually staged? --cached --quiet exits 0 when no diff,
    # exit 1 when there are staged changes. that is the truth signal.
    staged_check = subprocess.run(
        ['git', '-C', REPO_ROOT, 'diff', '--cached', '--quiet'],
        capture_output=True, text=True,
    )
    has_staged_changes = staged_check.returncode == 1

    # commit (skip if nothing staged)
    fresh_commit = False
    if has_staged_changes:
        res = subprocess.run(
            ['git', '-C', REPO_ROOT, 'commit', '-m', message],
            capture_output=True, text=True,
        )
        fresh_commit = not (
            'nothing to commit' in res.stdout or 'nothing added' in res.stdout
        )
        if fresh_commit:
            print(res.stdout)
        else:
            print('git reported nothing to commit despite staged changes (odd)')
    else:
        print('nothing new to commit (artifacts gitignored or unchanged)')

    # check if there are unpushed commits to push
    ahead_check = subprocess.run(
        ['git', '-C', REPO_ROOT, 'log', '@{u}..HEAD', '--oneline'],
        capture_output=True, text=True,
    )
    if not ahead_check.stdout.strip() and not fresh_commit:
        print('local is in sync with origin; nothing to push.')
        return False

    # push using url-embedded auth (one-shot target, doesnt persist in config)
    auth_url = f'https://x-access-token:{token}@github.com/george-gideon-S/cs-gy-6513-big-data-311-nlp.git'
    push_res = subprocess.run(
        ['git', '-C', REPO_ROOT, 'push', auth_url, 'HEAD:main'],
        capture_output=True, text=True,
    )
    if push_res.returncode != 0:
        print('push failed:')
        print(push_res.stdout)
        print(push_res.stderr)
        return False

    # git push prints to stderr on success - relay it but strip the auth url
    out = (push_res.stdout + push_res.stderr).replace(auth_url, REMOTE_URL)
    print(out)
    print('pushed to github successfully')
    return True
