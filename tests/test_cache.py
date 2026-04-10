"""
tests/test_cache.py
--------------------
Unit tests for the PaperCache utility.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from app.utils.cache import PaperCache


class TestPaperCache:
    def setup_method(self):
        # Use a temporary file for each test
        self._tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self._tmpdir, "test_cache.json")
        self.cache = PaperCache(cache_file=self.cache_file)

    def test_unseen_paper_is_not_in_cache(self):
        assert not self.cache.is_seen("2401.12345")

    def test_mark_seen_persists(self):
        self.cache.mark_seen("2401.12345")
        # Reload from disk
        cache2 = PaperCache(cache_file=self.cache_file)
        assert cache2.is_seen("2401.12345")

    def test_unseen_ids_filters_correctly(self):
        self.cache.mark_seen("seen-001")
        ids = ["seen-001", "new-002", "new-003"]
        unseen = self.cache.unseen_ids(ids)
        assert unseen == ["new-002", "new-003"]

    def test_handles_corrupt_cache_file(self):
        with open(self.cache_file, "w") as f:
            f.write("NOT VALID JSON {{")
        # Should not raise; starts fresh
        cache = PaperCache(cache_file=self.cache_file)
        assert not cache.is_seen("any-id")

    def test_empty_unseen_ids(self):
        unseen = self.cache.unseen_ids([])
        assert unseen == []
