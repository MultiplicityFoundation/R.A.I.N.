## 2025-02-14 - Pre-computing Lowercase Content
**Learning:** Text lowercasing for fuzzy matching in a loop is expensive (O(N * K)). Pre-computing it during load (O(N)) yields significant speedups (5x) for frequent lookups.
**Action:** When performing repeated case-insensitive searches on static content, always cache the normalized version.
