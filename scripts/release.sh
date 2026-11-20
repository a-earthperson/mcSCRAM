#!/usr/bin/env bash
# Cut a release: set the version everywhere, commit it to develop, tag it,
# and push both to origin in one atomic push.
#
#   scripts/release.sh [options] VERSION|patch|minor|major
#
#   VERSION     X.Y.Z, greater than the current version
#   patch|minor|major
#               bump the current version
#   -n, --dry-run
#               rewrite the files and show the diff; no commit, tag or push
#   --no-push   commit and tag locally; print the push command
#   -y, --yes   push without asking
#   -h, --help  show this help
#
# Runs from a clean checkout of develop that matches origin/develop. The push
# starts release.yml (PyPI, GitHub Release; approve the pypi and pypi-server
# deployments) and server-image.yml (live.fuzzybit.org). See docs/releasing.md.
set -euo pipefail

usage() {
  sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "$0"
}

die() {
  echo "release: $*" >&2
  exit 1
}

dry_run=false
push=true
assume_yes=false
target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n | --dry-run) dry_run=true ;;
    --no-push) push=false ;;
    -y | --yes) assume_yes=true ;;
    -h | --help) usage; exit 0 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *)
      [[ -z ${target} ]] || die "one version only (see --help)"
      target="$1"
      ;;
  esac
  shift
done
[[ -n ${target} ]] || { usage >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"
remote=origin
branch=develop

old="$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml | head -1)"
[[ ${old} =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "current version '${old}' in pyproject.toml is not X.Y.Z"
major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}" patch="${BASH_REMATCH[3]}"
case "${target}" in
  patch) new="${major}.${minor}.$((patch + 1))" ;;
  minor) new="${major}.$((minor + 1)).0" ;;
  major) new="$((major + 1)).0.0" ;;
  *) new="${target#v}" ;;
esac
# CMake's project(VERSION) and PyPI both need a plain X.Y.Z here.
[[ ${new} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version '${new}' is not X.Y.Z"
[[ "$(printf '%s\n%s\n' "${old}" "${new}" | sort -V | tail -1)" == "${new}" && ${new} != "${old}" ]] \
  || die "${new} is not greater than the current ${old}"
tag="v${new}"

# Preconditions: develop, clean, in step with origin, tag unused.
[[ "$(git symbolic-ref --quiet --short HEAD || true)" == "${branch}" ]] || die "check out ${branch} first"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "working tree has uncommitted changes"
git fetch --quiet --tags "${remote}" "${branch}"
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${remote}/${branch}")" ]] \
  || die "${branch} is not at ${remote}/${branch}; pull or push first"
! git rev-parse --quiet --verify "refs/tags/${tag}" >/dev/null || die "tag ${tag} already exists"
[[ -z "$(git ls-remote --tags "${remote}" "refs/tags/${tag}")" ]] || die "tag ${tag} already exists on ${remote}"

echo "release: ${old} -> ${new} (${tag}) on ${branch} at $(git rev-parse --short HEAD)"

# Each file and the edit that moves it from old to new. Every file must
# change, so an edit whose pattern no longer matches fails the release
# instead of leaving a stale version behind.
export OLD="${old}" NEW="${new}"
edit() {
  local file=$1 expr=$2
  [[ -f ${file} ]] || die "${file} is missing"
  perl -pi -e "${expr}" "${file}"
  git diff --quiet -- "${file}" && die "${file}: no version string matched; update scripts/release.sh"
  # other packages in a lock file can share the version string
  if [[ ${file} != *.lock ]] && grep -qF "${old}" "${file}"; then
    grep -nF "${old}" "${file}" >&2
    die "${file} still mentions ${old}"
  fi
}
# lock files: only the version line of our own package entries
lock_edit='$p = $1 if /^name = "([^"]*)"/; s/^version = "\Q$ENV{OLD}\E"/version = "$ENV{NEW}"/ if $p =~ /^fuzzybits(-server)?$/'

edit pyproject.toml 's/^version = "\Q$ENV{OLD}\E"/version = "$ENV{NEW}"/'
edit python/fuzzybits/__init__.py 's/^__version__ = "\Q$ENV{OLD}\E"/__version__ = "$ENV{NEW}"/'
edit tools/server/pyproject.toml 's/^version = "\Q$ENV{OLD}\E"/version = "$ENV{NEW}"/; s/"fuzzybits==\Q$ENV{OLD}\E"/"fuzzybits==$ENV{NEW}"/'
edit CMakeLists.txt 's/^project\(SCRAM VERSION \Q$ENV{OLD}\E /project(SCRAM VERSION $ENV{NEW} /'
edit .env 's/^FUZZYBITS_SERVER_TAG=v\Q$ENV{OLD}\E$/FUZZYBITS_SERVER_TAG=v$ENV{NEW}/'
edit README.md 's/alpha \(\Q$ENV{OLD}\E\)/alpha ($ENV{NEW})/; s/fuzzybits-server:v\Q$ENV{OLD}\E\b/fuzzybits-server:v$ENV{NEW}/g'
edit docker/runtime.Dockerfile 's/^ARG FUZZYBITS_VERSION=\Q$ENV{OLD}\E$/ARG FUZZYBITS_VERSION=$ENV{NEW}/'
edit docker/wheel-fallback.Dockerfile 's/fuzzybits-\Q$ENV{OLD}\E-cp/fuzzybits-$ENV{NEW}-cp/g'
edit tools/server/Dockerfile 's/^ARG (FUZZYBITS|SERVER)_VERSION=\Q$ENV{OLD}\E$/ARG $1_VERSION=$ENV{NEW}/'
edit tools/server/README.md 's/fuzzybits==\Q$ENV{OLD}\E\b/fuzzybits==$ENV{NEW}/g; s/fuzzybits_server-\Q$ENV{OLD}\E-/fuzzybits_server-$ENV{NEW}-/g'
edit uv.lock "${lock_edit}"
edit tools/server/uv.lock "${lock_edit}"

bash scripts/ci/release-version.sh check "${tag}" >/dev/null
if command -v uv >/dev/null; then
  uv lock --check --quiet || die "uv.lock is out of date after the bump"
  (cd tools/server && uv lock --check --quiet) || die "tools/server/uv.lock is out of date after the bump"
else
  echo "release: uv not found; skipped uv lock --check" >&2
fi

# Anything else that still names the old version (changelogs, prose) is
# reported, not rewritten: it may be deliberate.
if stale="$(git grep -nIwF "${old}" -- . ':!*.lock' ':!**/package-lock.json')"; then
  echo "release: other mentions of ${old}, left as they are:" >&2
  printf '%s\n' "${stale}" >&2
fi

git --no-pager diff --stat
if ${dry_run}; then
  git --no-pager diff
  echo "release: dry run; restore with: git checkout -- ."
  exit 0
fi

git commit --quiet --all -m "chore(release): ${new}"
git tag -a "${tag}" -m "fuzzybits ${new}"
echo "release: committed $(git rev-parse --short HEAD) and tagged ${tag}"

push_cmd=(git push --atomic "${remote}" "${branch}" "refs/tags/${tag}")
if ! ${push}; then
  echo "release: not pushed; run: ${push_cmd[*]}"
  exit 0
fi
if ! ${assume_yes}; then
  read -r -p "Push ${branch} and ${tag} to ${remote}? [y/N] " answer
  if [[ ! ${answer} =~ ^[Yy]$ ]]; then
    echo "release: not pushed; run: ${push_cmd[*]}"
    echo "release: or undo: git tag -d ${tag} && git reset --hard ${remote}/${branch}"
    exit 0
  fi
fi
"${push_cmd[@]}"
echo "release: pushed ${tag}; approve the pypi and pypi-server deployments in the Release run"
