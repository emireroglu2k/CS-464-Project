"""Trim dataset by selecting top-N train classes by image count.

Creates a destination folder (default: `trimmed dataset`) with a `train/`
subfolder containing only the top-N classes (copied from `datasets/train`).

Usage examples:
  python preprocess.py --preview
  python preprocess.py --top 10
  python preprocess.py --top 5 --dest "trimmed dataset" --force

The script counts files with common image extensions.
"""

from pathlib import Path
import shutil
import argparse
import sys

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def count_images_in_dir(path: Path) -> int:
	if not path.exists() or not path.is_dir():
		return 0
	count = 0
	for p in path.iterdir():
		if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
			count += 1
	return count


def get_class_counts(train_root: Path):
	classes = []
	if not train_root.exists() or not train_root.is_dir():
		raise FileNotFoundError(f"Train folder not found: {train_root}")
	for class_dir in sorted(train_root.iterdir()):
		if class_dir.is_dir():
			cnt = count_images_in_dir(class_dir)
			classes.append((class_dir.name, cnt, class_dir))
	# sort by count descending
	classes.sort(key=lambda x: x[1], reverse=True)
	return classes


def copy_top_classes(classes, top_n: int, dest_root: Path, force: bool = False, dataset_root: Path = None, copy_test: bool = False):
	dest_train = dest_root / 'train'
	if dest_root.exists() and not force:
		raise FileExistsError(f"Destination '{dest_root}' already exists. Use --force to overwrite.")
	if dest_root.exists() and force:
		shutil.rmtree(dest_root)
	dest_train.mkdir(parents=True, exist_ok=True)

	dest_test = None
	if copy_test:
		dest_test = dest_root / 'test'
		dest_test.mkdir(parents=True, exist_ok=True)
		if dataset_root is None:
			raise ValueError("dataset_root must be provided when copy_test is True")

	selected = classes[:top_n]
	for name, cnt, src_path in selected:
		# copy train files
		dst_class = dest_train / name
		dst_class.mkdir(parents=True, exist_ok=True)
		copied_train = 0
		for p in src_path.iterdir():
			if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
				shutil.copy2(p, dst_class / p.name)
				copied_train += 1
		print(f"Copied {copied_train} images for class '{name}' -> {dst_class}")

		# copy test files if requested
		if copy_test:
			src_test_class = dataset_root / 'test' / name
			dst_test_class = dest_test / name
			if src_test_class.exists() and src_test_class.is_dir():
				dst_test_class.mkdir(parents=True, exist_ok=True)
				copied_test = 0
				for p in src_test_class.iterdir():
					if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
						shutil.copy2(p, dst_test_class / p.name)
						copied_test += 1
				print(f"Copied {copied_test} test images for class '{name}' -> {dst_test_class}")
			else:
				print(f"No test folder for class '{name}' at {src_test_class}; skipping test copy")


def build_parser():
	p = argparse.ArgumentParser(description="Trim dataset to top-N train classes by image count")
	p.add_argument('--dataset', default='datasets', help='Path to the dataset root (default: datasets)')
	p.add_argument('--top', type=int, default=10, help='Number of top classes to keep (default: 10)')
	p.add_argument('--dest', default='trimmed dataset', help='Destination folder (default: "trimmed dataset")')
	p.add_argument('--preview', action='store_true', help='Only print class counts and top-N; do not copy')
	p.add_argument('--force', action='store_true', help='Overwrite destination if it exists')
	p.add_argument('--copy-test', action='store_true', help='Also copy corresponding class folders from the dataset test/ directory')
	return p


def main(argv=None):
	argv = argv if argv is not None else sys.argv[1:]
	parser = build_parser()
	args = parser.parse_args(argv)

	dataset_root = Path(args.dataset)
	train_root = dataset_root / 'train'
	dest_root = Path(args.dest)

	try:
		classes = get_class_counts(train_root)
	except FileNotFoundError as e:
		print(e)
		return 2

	if not classes:
		print(f"No class directories found under {train_root}")
		return 1

	print("Found classes (sorted by image count):")
	for i, (name, cnt, _) in enumerate(classes, start=1):
		print(f"{i:2d}. {name}: {cnt}")

	top_n = min(args.top, len(classes))
	print(f"\nTop {top_n} classes to keep:")
	for i, (name, cnt, _) in enumerate(classes[:top_n], start=1):
		print(f"{i:2d}. {name}: {cnt}")

	if args.preview:
		print("\nPreview mode: no files were copied. Use without --preview to perform copying.")
		return 0

	# perform copy
	try:
		copy_top_classes(classes, top_n, dest_root, force=args.force, dataset_root=dataset_root, copy_test=args.copy_test)
	except FileExistsError as e:
		print(e)
		return 3

	print(f"\nDone. Trimmed dataset created at: {dest_root.resolve()}")
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

