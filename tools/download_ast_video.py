import os


def load_txt_file(f_path):
    if not os.path.isfile(f_path):
        raise Exception(f"{f_path} does not exist, please check your path")
    else:
        with open(f_path, 'r') as f:
            f_data = f.readlines()
        return f_data


def extract_url_from_txt_data(f_path):
    # load target seq
    seq_list = load_txt_file(f_path=f_path)

    # load all the url

    print(f"this is the end of func")


def main():
    val_list_path = os.path.join('ast_data', 'pose_val.txt')
    # test_list_path = os.path.join('ast_data', 'pose_test.txt')

    extract_url_from_txt_data(f_path=val_list_path)


    print(f" this is the end")


if __name__ == '__main__':
    main()