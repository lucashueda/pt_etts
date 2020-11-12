import argparse

print('oi')

if __name__ == '__main__':
    """Run preprocessing process."""

    print('oi2')
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save mel files')
    args = parser.parse_args()

    print('oi')