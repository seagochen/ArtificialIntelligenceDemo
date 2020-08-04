import requests
import threading

BASE_URL = "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq"

DATA_SET_URL = "http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html"

FILES = [
    'Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl',
    'Bolt', 'Box', 'Car1', 'Car4', 'CarDark', 'CarScale', 'ClifBar',
    'Couple', 'Crowds', 'David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
    'Football', 'Freeman4', 'Girl', 'Human3', 'Human4', 'Human6', 'Human9',
    'Ironman', 'Jump', 'Jumping', 'Liquor', 'Matrix', 'MotorRolling', 'Panda',
    'RedTeam', 'Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing', 'Soccer',
    'Surfer', 'Sylvester', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman',
    'Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board', 'Bolt2', 'Boy',
    'Car2', 'Car24', 'Coke', 'Coupon', 'Crossing', 'Dancer', 'Dancer2',
    'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2',
    'Fish', 'FleetFace', 'Football1', 'Freeman1', 'Freeman3', 'Girl2', 'Gym',
    'Human2', 'Human5', 'Human7', 'Human8', 'Jogging', 'KiteSurf', 'Lemming',
    'Man', 'Mhyang', 'MountainBike', 'Rubik', 'Singer1', 'Skater', 'Skater2',
    'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans', 'Twinnings', 'Vase'
]

EXTENSIONS = [
    'zip', 'rar', '7z'
]


class DownloadFileThread(threading.Thread):

    def __init__(self, base_url, file_name):
        threading.Thread.__init__(self)
        self.base_url = base_url
        self.file_name = file_name

    def download_file(self, extension: str) -> bool:
        # generate completed file url
        completed_file_url = self.base_url + '/' + self.file_name + '.' + extension

        try:

            response = requests.get(completed_file_url, allow_redirects=True)

            if response.status_code == 200:  # source found
                open(self.file_name + '.' + extension, 'wb').write(response.content)
                print(f"file {self.file_name}.{extension} download success")
                return True

            else:
                return False

        except:

            return False

    def try_download_file(self, base_url: str, file_name: str):
        for extension in EXTENSIONS:

            print(f"trying to download file {file_name}.{extension}")

            if self.download_file(extension):
                break

            else:
                print(f"try {file_name}.{extension} failed, try another extension again")

        print(f"{file_name} is finally failed, go to the original website and download it by yourself. {DATA_SET_URL}")

    def run(self):
        self.try_download_file(self.base_url, self.file_name)

    def __del__(self):
        print(f"thread for {self.file_name} is finished")


if __name__ == "__main__":

    thread_list = []

    for name in FILES:
        thread = DownloadFileThread(BASE_URL, name)

        if len(thread_list) < 10:
            thread.start()
            thread_list.append(thread)

        else:
            for t in thread_list:
                t.join()

            thread_list = []

    print("download finished....")
