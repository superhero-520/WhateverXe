"""Exercise the offline override CLI with a synthetic EDID and a 120 Hz timing."""

import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "mne007_refresh_override.py"
TYPE_VII = bytes.fromhex("bfa10a85 3f0b 9f00 2f00 1f00 0707 6d00 0500 0500")
PROFILE = {"DisplayVendorID": 0xe77, "DisplayProductID": 0x1431, "scale-resolutions": [b"scaling"]}


def sample(descriptor=TYPE_VII):
    base, extension = bytearray(128), bytearray(128)
    base[:12] = bytes.fromhex("00ffffffffffff00 0e773114")
    base[18:21] = bytes.fromhex("0104b5")
    base[126] = 1
    base[127] = -sum(base) % 256
    extension[:8] = bytes.fromhex("7020170200 220014")
    extension[8:28] = descriptor
    extension[28] = -sum(extension[1:28]) % 256
    extension[127] = -sum(extension) % 256
    return bytes(base + extension)


def main():
    with tempfile.TemporaryDirectory() as work:
        folder = Path(work)
        edid_path, profile_path = folder / "panel.edid", folder / "current.plist"

        def run(edid, profile, name, success):
            edid_path.write_bytes(edid)
            profile_path.write_bytes(plistlib.dumps(profile))
            output = folder / name
            result = subprocess.run([sys.executable, str(TOOL), str(edid_path), str(profile_path), str(output)],
                                    capture_output=True)
            assert (result.returncode == 0) == success, result.stderr
            assert edid_path.read_bytes() == edid and plistlib.loads(profile_path.read_bytes()) == profile
            if success:
                return plistlib.loads(output.read_bytes())
            assert not output.exists()

        profile = run(sample(), PROFILE, "candidate", True)
        assert {k: v for k, v in profile.items() if k != "tspc"} == PROFILE
        record, = profile["tspc"]
        # Independently specified IOKit record with identical nominal/min/max clock,
        # blanking and sync. Big endian, no scale, format/depth or DSC request.
        clock = (696768000).to_bytes(8, "big")
        values = (2880, 160, 48, 32, 1800, 110, 6, 6)
        assert record == bytes(40) + clock * 3 + b"".join(n.to_bytes(4, "big") for n in values) + bytes(64)
        assert run(sample(), profile, "rerun", True) == profile
        prior = dict(PROFILE, tspc=[bytes(160)], IODisplayEDID=sample())
        assert run(sample(), prior, "merged", True) == dict(prior, tspc=[bytes(160), record])

        # Sign bits belong to horizontal/vertical sync offsets, not their magnitudes.
        positive = bytearray(TYPE_VII)
        positive[9] |= 0x80
        positive[17] |= 0x80
        pos_record, = run(sample(positive), PROFILE, "positive", True)["tspc"]
        expected = bytearray(record)
        expected[115] = expected[123] = 1
        assert pos_record == expected

        bad_section = bytearray(sample())
        bad_section[128 + 28] ^= 1
        bad_section[-1] = -sum(bad_section[128:-1]) % 256
        run(bad_section, PROFILE, "bad-section", False)
        run(sample()[:-128], PROFILE, "missing-block", False)
        run(sample(), dict(PROFILE, DisplayProductID=0x4321), "wrong-display", False)
        run(sample(), dict(PROFILE, IODisplayEDID=b"other"), "conflicting-edid", False)
        run(sample(), dict(PROFILE, tspc=[bytes(40)]), "invalid-original", False)
        wrong_rate = bytes([0]) + TYPE_VII[1:]
        run(sample(wrong_rate), PROFILE, "unadvertised", False)
    print("Offline MNE007ZA3-2 timing, merge and rejection cases passed.")


if __name__ == "__main__":
    main()
