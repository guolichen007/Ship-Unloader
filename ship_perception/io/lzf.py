"""Bounded LZF decoding for PCL field-major compressed PCD payloads."""


def decompress(payload, expected_size):
    output = bytearray(expected_size)
    source = memoryview(payload)
    i = written = 0
    while i < len(source):
        control = source[i]
        i += 1
        if control < 32:
            length = control + 1
            if i + length > len(source) or written + length > expected_size:
                raise ValueError("INVALID_PCD_LZF_LITERAL")
            output[written : written + length] = source[i : i + length]
            i += length
            written += length
        else:
            length = control >> 5
            offset = (control & 31) << 8
            if length == 7:
                if i >= len(source):
                    raise ValueError("TRUNCATED_PCD_LZF_REFERENCE")
                length += source[i]
                i += 1
            if i >= len(source):
                raise ValueError("TRUNCATED_PCD_LZF_REFERENCE")
            offset += source[i] + 1
            i += 1
            length += 2
            if offset > written or written + length > expected_size:
                raise ValueError("INVALID_PCD_LZF_REFERENCE")
            # Overlapping references repeat their preceding byte sequence.
            start = written - offset
            if length <= offset:
                output[written : written + length] = output[start : start + length]
            else:
                pattern = output[start:written]
                output[written : written + length] = (
                    pattern * ((length + offset - 1) // offset)
                )[:length]
            written += length
    if written != expected_size:
        raise ValueError("PCD_LZF_DECOMPRESSED_SIZE_MISMATCH")
    return output
