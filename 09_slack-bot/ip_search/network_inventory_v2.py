from __future__ import annotations

import csv
import ipaddress
import platform
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup


# ============================================================
# ファイル
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Advanced IP Scannerから出力したHTML
RESULTS_HTML = BASE_DIR / "results.html"

# 今回監視する端末一覧
MONITOR_TARGETS_CSV = BASE_DIR / "monitor_targets.csv"

# 除外された端末一覧
EXCLUDED_TARGETS_CSV = BASE_DIR / "excluded_targets.csv"

# ONLINE/OFFLINEの変化履歴
STATUS_CHANGE_CSV = BASE_DIR / "status_changes.csv"

# テキスト形式の履歴
STATUS_CHANGE_LOG = BASE_DIR / "status_changes.log"


# ============================================================
# 監視設定
# ============================================================

# 全対象の確認後、次の確認まで待つ時間
CHECK_INTERVAL_SECONDS = 10

# Windows pingコマンドのタイムアウト
PING_TIMEOUT_MS = 1500

# 連続失敗でOFFLINEと確定する回数
OFFLINE_THRESHOLD = 3

# 連続成功でONLINEと確定する回数
ONLINE_THRESHOLD = 2

# Pingを並列実行する最大数
MAX_WORKERS = 24

# 何秒ごとに現在状態の一覧を表示するか
SUMMARY_INTERVAL_SECONDS = 60


# ============================================================
# 名前による除外条件
# ============================================================

# 名前に含まれていれば除外する文字列
#
# 必要に応じて追加してください。
EXCLUDED_NAME_KEYWORDS = {
    "YUICHI",
    "MUTO",
    "MATSUNO",
    "KIMURA",
    "OKADA",
    "TAHARA",
    "SHINDAI",
    "GIJUTSU",
    "SCANNER",
    "MOUSE",
    "MACBOOK",
}

# ノートPCらしい名前
EXCLUDED_NAME_PATTERNS = [
    r"^LAPTOP-",
]

# ネットワーク機器や明らかな周辺機器
EXCLUDED_EQUIPMENT_NAME_KEYWORDS = {
    "PROCURVE",
    "SWITCH",
    "ROUTER",
    "AIR STATION",
    "AIRSTATION",
    "ACCESS POINT",
    "PRINTER",
}


# ============================================================
# メーカーによる分類
# ============================================================

# HP製PC・機器候補として扱うメーカー
HP_VENDOR_KEYWORDS = {
    "HP INC",
    "HEWLETT PACKARD",
    "HEWLETT-PACKARD",
}

# HP製だが、監視対象PCではない機器
HP_INFRASTRUCTURE_VENDOR_KEYWORDS = {
    "PROCURVE NETWORKING BY HP",
}

# HP製PCではないと比較的明確に判断できるメーカー
#
# これらは自動除外します。
EXCLUDED_VENDOR_KEYWORDS = {
    "APPLE",
    "BUFFALO",
    "DELL",
    "SIGMA KOKI",
    "RASPBERRY PI",
    "ASUSTEK",
    "ASROCK",
    "FUJITSU",
    "TOSHIBA",
    "CLEVO",
    "SEIKO EPSON",
    "CANON",
    "I-O DATA",
    "ELECOM",
    "PANASONIC",
    "COGNEX",
    "FUJIFILM",
    "SILEX TECHNOLOGY",
    "ELITEGROUP COMPUTER SYSTEMS",
}

# PC本体ではなくLAN部品やOEM製造元として表示される可能性があるため、
# 除外せず「不明」として監視するメーカー
AMBIGUOUS_VENDOR_KEYWORDS = {
    "INTEL",
    "REALTEK",
    "HON HAI",
    "FOXCONN",
    "CLOUD NETWORK TECHNOLOGY",
    "LITEON",
    "CHONGQING FUGUI",
    "WISTRON",
    "QUANTA",
    "LCFC",
    "ASKEY",
}


# ============================================================
# データ構造
# ============================================================

@dataclass
class Device:
    original_status: str
    name: str
    ip_address: str
    vendor: str
    mac_address: str
    category: str = ""
    reason: str = ""


@dataclass
class MonitorState:
    status: str = "UNKNOWN"
    success_count: int = 0
    failure_count: int = 0
    last_change: str = "-"
    last_response: str = "-"


# ============================================================
# 共通処理
# ============================================================

def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_upper(value: str) -> str:
    return normalize_text(value).upper()


def normalize_mac(mac_address: str) -> str:
    cleaned = re.sub(r"[^0-9A-Fa-f]", "", mac_address)

    if len(cleaned) != 12:
        return ""

    return ":".join(
        cleaned[index:index + 2].upper()
        for index in range(0, 12, 2)
    )


def valid_ipv4(ip_address: str) -> bool:
    try:
        ipaddress.IPv4Address(ip_address)
        return True
    except ipaddress.AddressValueError:
        return False


def contains_any(text: str, keywords: Set[str]) -> Optional[str]:
    upper_text = normalize_upper(text)

    for keyword in sorted(keywords):
        if keyword.upper() in upper_text:
            return keyword

    return None


# ============================================================
# 自分自身のIP取得
# ============================================================

def get_local_ipv4_addresses() -> Set[str]:
    """
    このWindows PC自身に割り当てられているIPv4アドレスを取得する。

    Advanced IP ScannerのHTMLに自分自身が含まれていても、
    監視対象には入れない。
    """
    addresses: Set[str] = {"127.0.0.1"}

    try:
        hostname = socket.gethostname()

        for result in socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_INET,
        ):
            ip_address = result[4][0]

            if valid_ipv4(ip_address):
                addresses.add(ip_address)

    except OSError:
        pass

    # ipconfigからも取得する
    if platform.system().lower() == "windows":
        try:
            result = subprocess.run(
                ["ipconfig"],
                capture_output=True,
                text=True,
                encoding="cp932",
                errors="replace",
                timeout=10,
                check=False,
            )

            pattern = re.compile(
                r"IPv4[^:]*:\s*"
                r"(\d{1,3}(?:\.\d{1,3}){3})",
                re.IGNORECASE,
            )

            for match in pattern.finditer(result.stdout):
                ip_address = match.group(1)

                if valid_ipv4(ip_address):
                    addresses.add(ip_address)

        except (OSError, subprocess.TimeoutExpired):
            pass

    return addresses


# ============================================================
# results.htmlの読み込み
# ============================================================

def load_devices_from_html(html_path: Path) -> List[Device]:
    if not html_path.exists():
        raise FileNotFoundError(
            f"results.htmlが見つかりません。\n"
            f"配置場所: {html_path}"
        )

    html_text = html_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    soup = BeautifulSoup(html_text, "html.parser")

    devices: List[Device] = []

    for table_row in soup.find_all("tr"):
        cells = table_row.find_all(
            "td",
            class_="head",
            recursive=False,
        )

        if len(cells) < 7:
            continue

        values = [
            normalize_text(cell.get_text(" ", strip=True))
            for cell in cells[:7]
        ]

        original_status = values[0]
        name = values[1]
        ip_address = values[2]
        vendor = values[3]
        mac_address = normalize_mac(values[4])

        if not valid_ipv4(ip_address):
            continue

        devices.append(
            Device(
                original_status=original_status,
                name=name,
                ip_address=ip_address,
                vendor=vendor,
                mac_address=mac_address,
            )
        )

    # 同一IPが複数回入っていた場合は最後の情報を採用する
    devices_by_ip: Dict[str, Device] = {}

    for device in devices:
        devices_by_ip[device.ip_address] = device

    result = list(devices_by_ip.values())

    result.sort(
        key=lambda item: ipaddress.ip_address(item.ip_address)
    )

    return result


# ============================================================
# 監視対象・除外対象の分類
# ============================================================

def classify_device(
    device: Device,
    local_ip_addresses: Set[str],
) -> Tuple[str, str]:
    """
    戻り値:
        category:
            HP_CANDIDATE
            UNKNOWN
            EXCLUDED

        reason:
            分類理由
    """

    name_upper = normalize_upper(device.name)
    vendor_upper = normalize_upper(device.vendor)

    # 自分自身
    if device.ip_address in local_ip_addresses:
        return "EXCLUDED", "監視プログラムを実行しているPC自身"

    if name_upper == "HOST.DOCKER.INTERNAL":
        return "EXCLUDED", "監視プログラムを実行しているPC自身"

    # ノートPC形式
    for pattern in EXCLUDED_NAME_PATTERNS:
        if re.search(pattern, name_upper, flags=re.IGNORECASE):
            return "EXCLUDED", f"ノートPC形式の名前: {device.name}"

    # 個人名、組織名、用途名
    matched_name = contains_any(
        device.name,
        EXCLUDED_NAME_KEYWORDS,
    )

    if matched_name:
        return (
            "EXCLUDED",
            f"除外対象の名前を含む: {matched_name}",
        )

    # ネットワーク機器や周辺機器
    matched_equipment = contains_any(
        device.name,
        EXCLUDED_EQUIPMENT_NAME_KEYWORDS,
    )

    if matched_equipment:
        return (
            "EXCLUDED",
            f"PC以外の機器名を含む: {matched_equipment}",
        )

    # HP ProCurveなど
    matched_hp_infrastructure = contains_any(
        vendor_upper,
        HP_INFRASTRUCTURE_VENDOR_KEYWORDS,
    )

    if matched_hp_infrastructure:
        return (
            "EXCLUDED",
            f"HP製だがネットワーク機器: {device.vendor}",
        )

    # HP候補
    matched_hp = contains_any(
        vendor_upper,
        HP_VENDOR_KEYWORDS,
    )

    if matched_hp:
        return (
            "HP_CANDIDATE",
            f"HP系メーカー: {device.vendor}",
        )

    # 明確な対象外メーカー
    matched_excluded_vendor = contains_any(
        vendor_upper,
        EXCLUDED_VENDOR_KEYWORDS,
    )

    if matched_excluded_vendor:
        return (
            "EXCLUDED",
            f"対象外メーカー: {device.vendor}",
        )

    # 部品メーカーやOEMメーカー
    matched_ambiguous_vendor = contains_any(
        vendor_upper,
        AMBIGUOUS_VENDOR_KEYWORDS,
    )

    if matched_ambiguous_vendor:
        return (
            "UNKNOWN",
            f"PC部品またはOEMメーカーの可能性: {device.vendor}",
        )

    # メーカー名なし
    if not vendor_upper:
        return (
            "UNKNOWN",
            "メーカー情報なし",
        )

    # DESKTOP形式なら念のため残す
    if name_upper.startswith("DESKTOP-"):
        return (
            "UNKNOWN",
            f"Windowsデスクトップ形式、メーカーは要確認: {device.vendor}",
        )

    # 知らないメーカーも誤除外を避けるため監視する
    return (
        "UNKNOWN",
        f"未分類メーカー: {device.vendor}",
    )


def classify_all_devices(
    devices: List[Device],
) -> Tuple[List[Device], List[Device]]:
    local_ip_addresses = get_local_ipv4_addresses()

    monitored: List[Device] = []
    excluded: List[Device] = []

    for device in devices:
        category, reason = classify_device(
            device,
            local_ip_addresses,
        )

        device.category = category
        device.reason = reason

        if category in {"HP_CANDIDATE", "UNKNOWN"}:
            monitored.append(device)
        else:
            excluded.append(device)

    return monitored, excluded


# ============================================================
# CSV出力
# ============================================================

def write_device_csv(
    output_path: Path,
    devices: List[Device],
) -> None:
    fieldnames = [
        "分類",
        "IPアドレス",
        "名前",
        "製造社",
        "MACアドレス",
        "Advanced IP Scanner状態",
        "分類理由",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for device in devices:
            writer.writerow(
                {
                    "分類": device.category,
                    "IPアドレス": device.ip_address,
                    "名前": device.name,
                    "製造社": device.vendor,
                    "MACアドレス": device.mac_address,
                    "Advanced IP Scanner状態":
                        device.original_status,
                    "分類理由": device.reason,
                }
            )


# ============================================================
# Ping監視
# ============================================================

def ping_host(ip_address: str) -> bool:
    """
    対象IPへPingを1回送信する。
    """
    if platform.system().lower() == "windows":
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

    except (subprocess.TimeoutExpired, OSError):
        return False


def ping_all_devices(
    devices: List[Device],
) -> Dict[str, bool]:
    """
    すべての監視対象へ並列でPingする。
    """
    results: Dict[str, bool] = {}

    worker_count = min(
        MAX_WORKERS,
        max(1, len(devices)),
    )

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        future_to_device = {
            executor.submit(
                ping_host,
                device.ip_address,
            ): device
            for device in devices
        }

        for future in as_completed(future_to_device):
            device = future_to_device[future]

            try:
                results[device.ip_address] = future.result()
            except Exception:
                results[device.ip_address] = False

    return results


# ============================================================
# 状態変化ログ
# ============================================================

def initialize_change_csv() -> None:
    if STATUS_CHANGE_CSV.exists():
        return

    with STATUS_CHANGE_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "時刻",
                "分類",
                "IPアドレス",
                "名前",
                "製造社",
                "MACアドレス",
                "変更前",
                "変更後",
            ]
        )


def write_status_change(
    device: Device,
    previous_status: str,
    new_status: str,
) -> None:
    timestamp = now_string()

    line = (
        f"{timestamp} | STATUS CHANGE | "
        f"{device.category} | "
        f"{device.ip_address} | "
        f"{device.name or '-'} | "
        f"{previous_status} -> {new_status}"
    )

    print(line, flush=True)

    with STATUS_CHANGE_LOG.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(line + "\n")

    initialize_change_csv()

    with STATUS_CHANGE_CSV.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                timestamp,
                device.category,
                device.ip_address,
                device.name,
                device.vendor,
                device.mac_address,
                previous_status,
                new_status,
            ]
        )


# ============================================================
# 状態表示
# ============================================================

def display_initial_targets(
    monitored: List[Device],
    excluded: List[Device],
) -> None:
    hp_count = sum(
        1
        for device in monitored
        if device.category == "HP_CANDIDATE"
    )

    unknown_count = sum(
        1
        for device in monitored
        if device.category == "UNKNOWN"
    )

    print()
    print("=" * 100)
    print(f"results.html読込時刻: {now_string()}")
    print(f"HP候補: {hp_count}台")
    print(f"不明端末: {unknown_count}台")
    print(f"監視対象合計: {len(monitored)}台")
    print(f"除外: {len(excluded)}台")
    print("=" * 100)

    for device in monitored:
        print(
            f"{device.category:<12} | "
            f"{device.ip_address:<15} | "
            f"{device.name or '-':<25} | "
            f"{device.vendor or '-':<45} | "
            f"{device.reason}"
        )

    print("=" * 100)
    print()


def display_summary(
    devices: List[Device],
    states: Dict[str, MonitorState],
) -> None:
    print()
    print(
        f"===== 現在状態 {now_string()} "
        f"監視対象={len(devices)}台 ====="
    )

    for device in devices:
        state = states[device.ip_address]

        print(
            f"{device.category:<12} "
            f"{device.ip_address:<15} "
            f"{state.status:<8} "
            f"{device.name or '-':<25} "
            f"最終変化={state.last_change}"
        )

    online_count = sum(
        state.status == "ONLINE"
        for state in states.values()
    )

    offline_count = sum(
        state.status == "OFFLINE"
        for state in states.values()
    )

    unknown_count = sum(
        state.status == "UNKNOWN"
        for state in states.values()
    )

    print(
        f"ONLINE={online_count} | "
        f"OFFLINE={offline_count} | "
        f"判定待ち={unknown_count}"
    )
    print("=" * 100)
    print()


# ============================================================
# メイン監視
# ============================================================

def monitor_forever(devices: List[Device]) -> None:
    states: Dict[str, MonitorState] = {
        device.ip_address: MonitorState()
        for device in devices
    }

    last_summary_time = 0.0

    print(
        f"{now_string()} | 監視開始 | "
        f"Targets={len(devices)} | "
        f"Interval={CHECK_INTERVAL_SECONDS}s | "
        f"OnlineThreshold={ONLINE_THRESHOLD} | "
        f"OfflineThreshold={OFFLINE_THRESHOLD}",
        flush=True,
    )

    try:
        while True:
            ping_results = ping_all_devices(devices)

            for device in devices:
                state = states[device.ip_address]

                reachable = ping_results.get(
                    device.ip_address,
                    False,
                )

                if reachable:
                    state.success_count += 1
                    state.failure_count = 0
                    state.last_response = now_string()

                    if (
                        state.status != "ONLINE"
                        and
                        state.success_count >= ONLINE_THRESHOLD
                    ):
                        previous_status = state.status
                        state.status = "ONLINE"
                        state.last_change = now_string()

                        write_status_change(
                            device,
                            previous_status,
                            "ONLINE",
                        )

                else:
                    state.failure_count += 1
                    state.success_count = 0

                    if (
                        state.status != "OFFLINE"
                        and
                        state.failure_count >= OFFLINE_THRESHOLD
                    ):
                        previous_status = state.status
                        state.status = "OFFLINE"
                        state.last_change = now_string()

                        write_status_change(
                            device,
                            previous_status,
                            "OFFLINE",
                        )

            current_monotonic = time.monotonic()

            if (
                current_monotonic - last_summary_time
                >= SUMMARY_INTERVAL_SECONDS
            ):
                display_summary(devices, states)
                last_summary_time = current_monotonic

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print()
        print(f"{now_string()} | 監視終了 | Ctrl+C")


def main() -> None:
    devices = load_devices_from_html(RESULTS_HTML)

    monitored, excluded = classify_all_devices(devices)

    write_device_csv(
        MONITOR_TARGETS_CSV,
        monitored,
    )

    write_device_csv(
        EXCLUDED_TARGETS_CSV,
        excluded,
    )

    display_initial_targets(
        monitored,
        excluded,
    )

    if not monitored:
        raise SystemExit(
            "監視対象が0台です。除外条件を確認してください。"
        )

    monitor_forever(monitored)


if __name__ == "__main__":
    main()