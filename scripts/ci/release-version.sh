#!/usr/bin/env bash
# Release versioning for .github/workflows/release.yml.
#
#   release-version.sh check [TAG]   verify that every version string agrees
#                                    (and equals TAG without its leading `v`);
#                                    print `version=` and `server=` for $GITHUB_OUTPUT
#   release-version.sh apply VERSION rewrite every version string to VERSION
#                                    (TestPyPI rehearsals: X.Y.Z -> X.Y.Z.devN)
#
# Version strings: pyproject.toml `version`, python/fuzzybits/__init__.py
# `__version__`, and, once tools/server is the `fuzzybits-server`
# distribution, its `version` plus the `fuzzybits==` pin.
set -euo pipefail

root_toml=pyproject.toml
init_py=python/fuzzybits/__init__.py
server_toml=tools/server/pyproject.toml

toml_version() {
  sed -n 's/^version = "\([^"]*\)"/\1/p' "$1" | head -1
}

server_ready() {
  grep -q '^name = "fuzzybits-server"' "${server_toml}"
}

die() {
  echo "::error::$*" >&2
  exit 1
}

cmd="${1:?usage: release-version.sh check [TAG] | apply VERSION}"
case "${cmd}" in
  check)
    tag="${2:-}"
    version="$(toml_version "${root_toml}")"
    [[ -n "${version}" ]] || die "no version in ${root_toml}"
    init_version="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${init_py}" | head -1)"
    [[ "${init_version}" == "${version}" ]] || die "${init_py} __version__=${init_version} != ${version}"
    server=false
    if server_ready; then
      server=true
      server_version="$(toml_version "${server_toml}")"
      pin="$(sed -n 's/.*"fuzzybits==\([^"]*\)".*/\1/p' "${server_toml}" | head -1)"
      [[ "${server_version}" == "${version}" ]] || die "${server_toml} version=${server_version} != ${version}"
      [[ "${pin}" == "${version}" ]] || die "${server_toml} pins fuzzybits==${pin}, expected ${version}"
    fi
    if [[ -n "${tag}" && "${tag}" != "v${version}" ]]; then
      die "tag ${tag} does not match version ${version} (expected v${version})"
    fi
    echo "version=${version}"
    echo "server=${server}"
    ;;
  apply)
    v="${2:?version}"
    # -i.bak: Windows perl cannot edit in place without a backup file.
    perl -pi.bak -e "s/^version = \"[^\"]*\"/version = \"${v}\"/" "${root_toml}"
    perl -pi.bak -e "s/^__version__ = \"[^\"]*\"/__version__ = \"${v}\"/" "${init_py}"
    if server_ready; then
      perl -pi.bak -e "s/^version = \"[^\"]*\"/version = \"${v}\"/; s/\"fuzzybits==[^\"]*\"/\"fuzzybits==${v}\"/" "${server_toml}"
    fi
    rm -f "${root_toml}.bak" "${init_py}.bak" "${server_toml}.bak"
    grep -HnE '^version = |^__version__ = |"fuzzybits==' "${root_toml}" "${init_py}" "${server_toml}" || true
    ;;
  *)
    die "unknown command ${cmd}"
    ;;
esac
