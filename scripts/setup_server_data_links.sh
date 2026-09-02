#!/usr/bin/env bash
set -euo pipefail

# Create the /data1 compatibility paths expected by the repository.
#
# Server layout:
#   /data1/qxh/tac_vla/tac_data
#   /data1/qxh/tac_vla/outputs
#
# Repository-visible layout after setup:
#   /data1/tac_data -> /data1/qxh/tac_vla/tac_data
#   /data1/outputs  -> /data1/qxh/tac_vla/outputs
#
# Usage:
#   sudo scripts/setup_server_data_links.sh
#   scripts/setup_server_data_links.sh --check
#   sudo scripts/setup_server_data_links.sh /another/storage/root

usage() {
    echo "Usage: $0 [--check] [server-data-root]" >&2
}

check_only=0
if [[ "${1:-}" == "--check" ]]; then
    check_only=1
    shift
fi

if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi

server_data_root="${1:-/data1/qxh/tac_vla}"
compat_root="${TAC_VLA_COMPAT_ROOT:-/data1}"

if [[ ! -d "$server_data_root" ]]; then
    echo "ERROR: server data root does not exist: $server_data_root" >&2
    exit 1
fi

if [[ ! -d "$compat_root" ]]; then
    echo "ERROR: compatibility root does not exist: $compat_root" >&2
    exit 1
fi

for child in tac_data outputs; do
    source_path="$server_data_root/$child"
    link_path="$compat_root/$child"

    if [[ ! -d "$source_path" ]]; then
        echo "ERROR: required directory does not exist: $source_path" >&2
        exit 1
    fi

    if [[ -L "$link_path" ]]; then
        actual_target="$(readlink -f -- "$link_path")"
        expected_target="$(readlink -f -- "$source_path")"
        if [[ "$actual_target" != "$expected_target" ]]; then
            echo "ERROR: $link_path points to $actual_target, expected $expected_target" >&2
            exit 1
        fi
        echo "OK: $link_path -> $actual_target"
        continue
    fi

    if [[ -e "$link_path" ]]; then
        echo "ERROR: refusing to replace existing path: $link_path" >&2
        exit 1
    fi

    if [[ $check_only -eq 1 ]]; then
        echo "MISSING: $link_path -> $source_path" >&2
        exit 1
    fi

    ln -s -- "$source_path" "$link_path"
    echo "CREATED: $link_path -> $source_path"
done

echo "Data-path mapping is ready. Existing /data1-based commands can be used unchanged."
