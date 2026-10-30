import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.fetch_assets import install_archive


class FetchAssetsTests(unittest.TestCase):
    def archive(self, base, name='assets/example.bin', content=b'scene'):
        archive = base / 'payload.zip'
        with zipfile.ZipFile(archive, 'w') as out:
            out.writestr(name, content)
        spec = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files_sha256': {name: hashlib.sha256(content).hexdigest()}}
        return archive, spec

    def test_install_and_repeat_preserve_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'checkout'; root.mkdir()
            archive, spec = self.archive(base)
            install_archive(archive, spec, root)
            install_archive(archive, spec, root)
            self.assertEqual((root / 'assets/example.bin').read_bytes(), b'scene')

    def test_rejects_archive_hash_mismatch_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'checkout'; root.mkdir()
            archive, spec = self.archive(base); spec['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'archive checksum'):
                install_archive(archive, spec, root)
            self.assertEqual(list(root.iterdir()), [])

    def test_rejects_member_hash_mismatch_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'checkout'; root.mkdir()
            archive, spec = self.archive(base)
            spec['files_sha256']['assets/example.bin'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'member checksum'):
                install_archive(archive, spec, root)
            self.assertEqual(list(root.iterdir()), [])

    def test_rejects_path_escape_and_existing_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'checkout'; root.mkdir()
            archive, spec = self.archive(base, '../escape.bin')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                install_archive(archive, spec, root)
            archive, spec = self.archive(base)
            outside = base / 'outside'; outside.mkdir()
            (root / 'assets').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                install_archive(archive, spec, root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_refuses_to_overwrite_modified_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'checkout'; root.mkdir()
            archive, spec = self.archive(base)
            (root / 'assets').mkdir(); target = root / 'assets/example.bin'
            target.write_bytes(b'user edit')
            with self.assertRaisesRegex(ValueError, 'modified'):
                install_archive(archive, spec, root)
            self.assertEqual(target.read_bytes(), b'user edit')
