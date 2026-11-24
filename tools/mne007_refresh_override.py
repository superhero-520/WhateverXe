#!/usr/bin/env python3
"""Build an offline macOS display override for the MNE007ZA3-2's native 120 Hz timing.

Usage: python3 tools/mne007_refresh_override.py panel.edid current.plist candidate.plist
"""

import argparse
import plistlib
import struct
from pathlib import Path


def panel_timing(edid):
    if len(edid) < 128 or edid[:8] != bytes.fromhex("00ffffffffffff00"):
        raise ValueError("expected a binary EDID")
    if len(edid) != (1 + edid[126]) * 128:
        raise ValueError("EDID extension count differs from the file length")
    if any(sum(edid[i:i + 128]) % 256 for i in range(0, len(edid), 128)):
        raise ValueError("invalid EDID block checksum")
    if edid[8:12] != bytes.fromhex("0e773114") or edid[18:21] != bytes.fromhex("0104b5"):
        raise ValueError("expected the MNE007ZA3-2 digital 10-bit eDP panel")

    candidates = set()
    for start in range(128, len(edid), 128):
        ext = edid[start:start + 128]
        if ext[:2] != b"\x70\x20":
            continue
        end = 5 + ext[2]
        if end > 126 or sum(ext[1:end + 1]) % 256:
            raise ValueError("invalid DisplayID section length or checksum")
        pos = 5
        while pos < end and any(ext[pos:end]):
            if end - pos < 3:
                raise ValueError("incomplete DisplayID data block header")
            tag, revision, length = ext[pos:pos + 3]
            pos += 3
            if pos + length > end:
                raise ValueError("DisplayID data block exceeds its section")
            if (tag, revision) == (0x22, 0):
                if length == 0 or length % 20:
                    raise ValueError("invalid Type VII timing block length")
                for offset in range(pos, pos + length, 20):
                    descriptor = ext[offset:offset + 20]
                    clock = (int.from_bytes(descriptor[:3], "little") + 1) * 1000
                    # VESA DisplayID 2.0 Type VII. Sync offsets carry polarity in bit 15.
                    ha, hb, ho, hw, va, vb, vo, vw = struct.unpack_from("<8H", descriptor, 4)
                    geometry = (ha + 1, hb + 1, (ho & 0x7fff) + 1, hw + 1,
                                va + 1, vb + 1, (vo & 0x7fff) + 1, vw + 1)
                    width, hblank, hoff, hsync, height, vblank, voff, vsync = geometry
                    if (width, height) != (2880, 1800) or descriptor[3] & 0x70:
                        continue
                    if clock != 120 * (width + hblank) * (height + vblank):
                        continue
                    if hoff + hsync > hblank or voff + vsync > vblank:
                        raise ValueError("sync exceeds the timing blanking interval")
                    candidates.add((clock, geometry, ho >> 15, vo >> 15))
            pos += length
    if len(candidates) != 1:
        raise ValueError("expected one distinct 2880x1800 progressive 120 Hz Type VII timing")
    return candidates.pop()


def make_override(edid, original):
    clock, geometry, hpositive, vpositive = panel_timing(edid)
    if (original.get("DisplayVendorID"), original.get("DisplayProductID")) != (0xe77, 0x1431):
        raise ValueError("input override has a different display identity")
    if original.get("IODisplayEDID", edid) != edid:
        raise ValueError("the existing EDID override differs from the source EDID")
    timings = original.get("tspc", [])
    if not isinstance(timings, list) or not all(isinstance(t, bytes) and len(t) == 160 for t in timings):
        raise ValueError("invalid existing tspc array")

    # Apple's IOKitUser IODisplayLib.c, InstallFromTimingOverride/TimingToHost.
    # Use the big-endian override ABI with Hz precision. Leave wire format selection
    # to the framebuffer and keep all reserved, scaling and optional fields zero.
    record = bytearray(160)
    struct.pack_into(">3Q8I", record, 40, clock, clock, clock, *geometry)
    struct.pack_into(">I", record, 112, hpositive)
    struct.pack_into(">I", record, 120, vpositive)
    result = dict(original)
    entry = bytes(record)
    result["tspc"] = list(timings) if entry in timings else [*timings, entry]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edid", type=Path, help="original binary panel EDID")
    parser.add_argument("original", type=Path, help="existing display override plist")
    parser.add_argument("output", type=Path, help="new candidate plist")
    args = parser.parse_args()
    try:
        original = plistlib.loads(args.original.read_bytes())
        if not isinstance(original, dict):
            raise ValueError("expected a display override dictionary")
        data = plistlib.dumps(make_override(args.edid.read_bytes(), original), sort_keys=False)
        with args.output.open("xb") as stream:
            stream.write(data)
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
