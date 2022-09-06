import time
import statistics


def mysleep_sec(sleep_time_sec):
    time_list = []

    sleep_time_ns = sleep_time_sec * 10 ** 9
    start = time.perf_counter_ns()

    # 現在の時刻と開始時の時刻の差が、スリープさせる時間より小さい場合ループ
    while 1:
        current_time = time.perf_counter_ns()
        time_list.append(current_time)

        if (current_time - start) > sleep_time_ns:
            break

    if len(time_list) > 1:
        new_list = []
        for i in range(len(time_list)):
            new_list[i] = time_list[i + 1] - time_list[i]
    else:
        new_list = 'Non '
        pass

    print(f'average loop time: {statistics.mean(new_list)}')


if __name__ == '__main__':
    shot_sec = 100 * 10 ** -6
    mysleep_sec(shot_sec)