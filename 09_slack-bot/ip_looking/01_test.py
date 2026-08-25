import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ==============================
# 設定
# ==============================

TARGETS = {
    # "装置PC": "10.165.200.46",
    # "Ubuntuダミー": "10.165.203.235",
    "HP候補1": "10.165.200.162",
    "HP候補2": "10.165.202.61",
}

# すべての対象を確認した後の待機時間
CHECK_INTERVAL = 10

# Pingのタイムアウト
PING_TIMEOUT_MS = 2000

# 連続失敗何回でOFFLINEと判断するか
OFFLINE_THRESHOLD = 3

# 連続成功何回でONLINEと判断するか
ONLINE_THRESHOLD = 2

# 状態変化履歴
LOG_FILE = Path(__file__).with_name("status_change.log")


def now_string() -> str:
    """現在時刻を文字列で返す。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(message: str) -> None:
    """状態変化を画面とログファイルに記録する。"""
    line = f"{now_string()} | {message}"

    print(line, flush=True)

    try:
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError as error:
        print(
            f"{now_string()} | ログ書き込みエラー | {error}",
            flush=True,
        )


def ping_host(ip_address: str) -> bool:
    """対象IPへ1回Pingし、成功したらTrueを返す。"""

    system_name = platform.system().lower()

    if system_name == "windows":
        command = [
            "ping",
            "-n",
            "1",
            "-w",
            str(PING_TIMEOUT_MS),
            ip_address,
        ]
    else:
        timeout_seconds = max(
            1,
            round(PING_TIMEOUT_MS / 1000),
        )

        command = [
            "ping",
            "-c",
            "1",
            "-W",
            str(timeout_seconds),
            ip_address,
        ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=(PING_TIMEOUT_MS / 1000) + 2,
            check=False,
        )

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False

    except OSError as error:
        write_log(
            f"PING ERROR | IP={ip_address} | {error}"
        )
        return False


def status_text(status) -> str:
    """内部状態を表示文字列に変換する。"""
    if status is True:
        return "ONLINE"

    if status is False:
        return "OFFLINE"

    return "UNKNOWN"


def main() -> None:
    monitor_states = {}

    for target_name, ip_address in TARGETS.items():
        monitor_states[ip_address] = {
            "name": target_name,
            "status": None,
            "success_count": 0,
            "failure_count": 0,
            "last_change": None,
        }

    write_log(
        f"監視開始 | Targets={len(TARGETS)}台 | "
        f"Interval={CHECK_INTERVAL}s | "
        f"OfflineThreshold={OFFLINE_THRESHOLD} | "
        f"OnlineThreshold={ONLINE_THRESHOLD}"
    )

    for target_name, ip_address in TARGETS.items():
        write_log(
            f"監視対象 | Name={target_name} | IP={ip_address}"
        )

    try:
        while True:
            cycle_time = now_string()

            for target_name, ip_address in TARGETS.items():
                state = monitor_states[ip_address]
                reachable = ping_host(ip_address)

                if reachable:
                    state["success_count"] += 1
                    state["failure_count"] = 0

                    # ONLINE確定
                    if (
                        state["status"] is not True
                        and state["success_count"] >= ONLINE_THRESHOLD
                    ):
                        previous_status = status_text(state["status"])

                        state["status"] = True
                        state["last_change"] = now_string()

                        write_log(
                            f"STATUS CHANGE | "
                            f"Name={target_name} | "
                            f"IP={ip_address} | "
                            f"{previous_status} -> ONLINE"
                        )

                else:
                    state["failure_count"] += 1
                    state["success_count"] = 0

                    # OFFLINE確定
                    if (
                        state["status"] is not False
                        and state["failure_count"] >= OFFLINE_THRESHOLD
                    ):
                        previous_status = status_text(state["status"])

                        state["status"] = False
                        state["last_change"] = now_string()

                        write_log(
                            f"STATUS CHANGE | "
                            f"Name={target_name} | "
                            f"IP={ip_address} | "
                            f"{previous_status} -> OFFLINE"
                        )

            # 1周期ごとに現在状態を一覧表示
            print()
            print(f"===== 現在状態 {cycle_time} =====")

            for target_name, ip_address in TARGETS.items():
                state = monitor_states[ip_address]

                last_change = (
                    state["last_change"]
                    if state["last_change"] is not None
                    else "-"
                )

                print(
                    f"{target_name:<12} "
                    f"{ip_address:<15} "
                    f"{status_text(state['status']):<7} "
                    f"最終変化: {last_change}"
                )

            print("==============================")
            print()

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        write_log("監視終了 | Ctrl+C")


if __name__ == "__main__":
    main()