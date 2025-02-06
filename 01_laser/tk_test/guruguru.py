import threading
import time
import sys

class TestClass:
    def __init__(self):
        self.work_ended = False

    def guruguru(self):
        while not self.work_ended:
            sys.stdout.write('\b' + '|')
            sys.stdout.flush()
            time.sleep(0.12)
            sys.stdout.write('\b' + '/')
            sys.stdout.flush()
            time.sleep(0.12)
            sys.stdout.write('\b' + '-')
            sys.stdout.flush()
            time.sleep(0.12)
            sys.stdout.write('\b' + '\\')
            sys.stdout.flush()
            time.sleep(0.12)
        sys.stdout.write('\b' + ' ')  # 最後に末尾1文字を空白で書き換える
        sys.stdout.flush()

    def work(self):
        time.sleep(3)
        self.work_ended = True

    def multithread_processing(self):
        print("Collecting package metadata:  ", end="")
        t1 = threading.Thread(target=self.guruguru)
        t2 = threading.Thread(target=self.work)
        t1.start()
        t2.start()
        t2.join()
        t1.join()
        sys.stdout.write("\b" + "done")
        sys.stdout.flush()

if __name__ == '__main__':
    tc = TestClass()
    tc.multithread_processing()
    print()  # 改行して終了
