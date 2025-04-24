import argparse
import os.path
import re
import openpyxl
from PIL import Image


def read_mappings_excel(excel_file):
    mappings = {}
    try:
        workbook = openpyxl.load_workbook(excel_file)
        sheet = workbook.active

        # Read each row and create a tuple of (current_name, changed_name)
        for row in sheet.iter_rows(min_row=890, max_row=2310, values_only=True):  # skip the header
            slide_name, slide_path, matched_HE_slide_name, matched_HE_path = row[:4]

            # no match
            if matched_HE_slide_name is None:
                continue

            # Regular expression to extract the desired part
            match = re.search(r'^\d+-\d+_\d+_\d+', slide_name)

            # Check if the pattern was found
            if match:
                dir_name = match.group()
                # print(f'dir_name = {dir_name}')
                mappings[dir_name] = {'slide_name': slide_name, 'slide_path': slide_path,
                                      'matched_HE_slide_name': matched_HE_slide_name, 'matched_HE_path': matched_HE_path}

    except Exception as e:
        print(f"Error reading Excel file {excel_file}: {e}")

    return mappings


def create_pair_dir(root_dir: str, output_dir: str, pair_dir_name: str, pair_dict: dict):
    pair_dir_path = os.path.join(output_dir, pair_dir_name)
    if not os.path.exists(pair_dir_path):
        os.makedirs(pair_dir_path, exist_ok=True)

    batch_dir_name = os.path.dirname(os.path.dirname(pair_dict.get('slide_path')))
    IHC_batch_path = os.path.join(root_dir, batch_dir_name)
    IHC_thumbs_dir = os.path.join(IHC_batch_path, 'thumbs')
    IHC_thumb_file = next(filter(lambda x: f'thumb_{pair_dir_name}' in x, os.listdir(IHC_thumbs_dir)))
    IHC_png_thumb_path = os.path.join(pair_dir_path, IHC_thumb_file.replace('.jpg', '.png'))
    IHC_png_thumb_copy_path = os.path.join(pair_dir_path, f"copy_{IHC_thumb_file.replace('.jpg', '.png')}")
    jpg_IHC_thumb = Image.open(os.path.join(IHC_thumbs_dir, IHC_thumb_file))
    jpg_IHC_thumb.save(IHC_png_thumb_path, 'PNG')
    jpg_IHC_thumb.save(IHC_png_thumb_copy_path, 'PNG')

    HE_batch_dir_name = os.path.dirname(os.path.dirname(pair_dict.get('matched_HE_path')))
    HE_batch_path = os.path.join(root_dir, HE_batch_dir_name)
    HE_thumbs_dir = os.path.join(HE_batch_path, 'thumbs')
    matching_thumb = pair_dict.get('matched_HE_slide_name').split('.')[0]
    # print(f'matching_thumb = {matching_thumb}')
    try:
        HE_thumb_file = next(filter(lambda x: f'thumb_{matching_thumb}' in x, os.listdir(HE_thumbs_dir)))
    except StopIteration as e:
        print(f'Thumb {matching_thumb} was not found, skipping')
        return 
    HE_png_thumb_path = os.path.join(pair_dir_path, HE_thumb_file.replace('.jpg', '.png'))
    jpg_HE_thumb = Image.open(os.path.join(HE_thumbs_dir, HE_thumb_file))
    jpg_HE_thumb.save(HE_png_thumb_path, 'PNG')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Create a directory with matching thumb pairs as png files, each pair "
                                                 "in its own subdirectory.")
    parser.add_argument('-r', '--root', required=True, help='Root directory to search for files.')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to write files to.')
    parser.add_argument('-m', '--mapping_file', required=True,
                        help='Excel file with current_name and changed_name columns.')

    print(f'os.cwd() = {os.getcwd()}')

    args = parser.parse_args()
    print(f'args = {args}')

    # Read mappings from the Excel file
    mappings = read_mappings_excel(args.mapping_file)
    # print(f'mappings = {mappings}')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for pair_dir_name, pair_dict in mappings.items():
        create_pair_dir(root_dir=args.root, output_dir=args.output_dir, pair_dir_name=pair_dir_name, pair_dict=pair_dict)


if __name__ == "__main__":
    main()