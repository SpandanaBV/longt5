def compression_ratio(original: str, summary: str) -> float:
    if not original.strip():
        return 0.0
    return len(summary.split()) / max(1, len(original.split()))


if __name__ == "__main__":
    print("Compression ratio ~30–40% is ideal")
