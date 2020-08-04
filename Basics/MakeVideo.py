import sys


def search_list(path: str, is_folder=False, pattern: str = "*") -> list:
    import os
    import re

    searched_results = []
    for item in os.listdir(path):
        if is_folder and os.path.isdir(item):

            if pattern == "*":
                searched_results.append(item)
            else:
                if re.match(pattern, str(item)):
                    searched_results.append(item)

        elif not is_folder and os.path.isfile(item):

            if pattern == "*":
                searched_results.append(item)
            else:
                if re.match(pattern, str(item)):
                    searched_results.append(item)

    # return to caller
    return searched_results


def generate_file_sequence(folder_path: str, pattern: str = r"\.jpg$") -> list:
    from siki.basics import FileUtils
    from siki.dstruct.BinaryTreeNode import BinaryTreeNode

    files = FileUtils.search_files(folder_path, pattern)
    root_tree = BinaryTreeNode(0)

    for jpg in files:
        root, leaf = FileUtils.root_leaf(jpg)

        # trim the .jpg
        leaf = leaf[:-4]

        # append the item to dictionary
        root_tree.search_append(int(leaf), jpg)

    # re balance and generate in order traversal
    root_tree = root_tree.re_balance()
    leaves = root_tree.in_order_traversal()

    # return the list
    final_list = []
    for leaf in leaves:
        final_list.append(leaf[1])

    return final_list[1:]


def generate_video(video_name, fps, img_list: list):
    import cv2

    if not isinstance(img_list, list) or len(img_list) <= 0:
        return None

    cv2_img = cv2.imread(img_list[0])
    height, width, layers = cv2_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_write = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Appending the images to the video one by one
    for image in img_list:
        video_write.write(cv2.imread(image))

    video_write.release()  # releasing the video generated


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Invalid source path")
        exit(0)

    # search folders
    folder_list = search_list(sys.argv[1], True, r"^[^.]\w+$")

    # 30 frames per second
    for folder in folder_list:
        generate_video(f"{folder}_30fps.avi", 30, generate_file_sequence(folder))
        generate_video(f"{folder}_20fps.avi", 20, generate_file_sequence(folder))
        generate_video(f"{folder}_1fps.avi", 1, generate_file_sequence(folder))
