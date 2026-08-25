import csv
import ipaddress
import platform
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# ============================================================
# 設定
# ============================================================

SCAN_NETWORKS = [
    "10.165.200.0/24",
    "10.165.201.0/24",
    "10.165.202.0/24",
    "10.165.203.0/24",
]

PING_TIMEOUT_MS = 500

# 同時Ping数
MAX_WORKERS = 32

# nbtstatのタイムアウト
NETBIOS_TIMEOUT_SECONDS = 2

OUTPUT_CSV = Path(__file__).with_name("network_inventory.csv")


# ============================================================
# メーカー判定用OUI
#
# 完全なメーカーDBではありません。
# スキャン結果を確認するための暫定判定です。
# ============================================================

KNOWN_OUI_VENDORS = {
    "00:2B:F5": "BUFFALO INC.",
    "00:24:A5": "BUFFALO INC.",
    "C0:18:03": "HP Inc.",
    "50:65:F3": "Hewlett Packard",
    "94:E7:0B": "Intel Corporate",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_mac(mac_address: str) -> str:
    """
    MACアドレスを AA:BB:CC:DD:EE:FF 形式に統一する。
    """
    cleaned = re.sub(r"[^0-9A-Fa-f]", "", mac_address)

    if len(cleaned) != 12:
        return ""

    return ":".join(
        cleaned[index:index + 2].upper()
        for index in range(0, 12, 2)
    )


def ping_host(ip_address: str) -> bool:
    """
    対象IPへPingを1回送信する。
    応答した場合はTrueを返す。
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


def read_arp_table() -> dict[str, str]:
    """
    WindowsのARPテーブルを取得する。

    戻り値:
        {
            "10.165.200.46": "D8:B3:2F:32:F5:97"
        }
    """
    try:
        result = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            text=True,
            encoding="cp932",
            errors="replace",
            timeout=10,
            check=False,
        )

    except (OSError, subprocess.TimeoutExpired):
        return {}

    entries: dict[str, str] = {}

    pattern = re.compile(
        r"^\s*"
        r"(\d{1,3}(?:\.\d{1,3}){3})"
        r"\s+"
        r"([0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2}){5})"
        r"\s+",
        re.IGNORECASE,
    )

    for line in result.stdout.splitlines():
        match = pattern.search(line)

        if not match:
            continue

        ip_address = match.group(1)
        mac_address = normalize_mac(match.group(2))

        if mac_address:
            entries[ip_address] = mac_address

    return entries


def get_dns_hostname(ip_address: str) -> str:
    """
    逆引きDNSからホスト名を取得する。
    """
    try:
        hostname, _, _ = socket.gethostbyaddr(ip_address)
        return hostname.rstrip(".")

    except (socket.herror, socket.gaierror, TimeoutError, OSError):
        return ""


def get_netbios_name(ip_address: str) -> str:
    """
    nbtstatを使ってWindowsのNetBIOS名を取得する。
    """
    if platform.system().lower() != "windows":
        return ""

    try:
        result = subprocess.run(
            ["nbtstat", "-A", ip_address],
            capture_output=True,
            text=True,
            encoding="cp932",
            errors="replace",
            timeout=NETBIOS_TIMEOUT_SECONDS,
            check=False,
        )

    except (OSError, subprocess.TimeoutExpired):
        return ""

    pattern = re.compile(
        r"^\s*([^\s<]+)\s*<00>\s+(?:一意|UNIQUE)",
        re.IGNORECASE,
    )

    for line in result.stdout.splitlines():
        match = pattern.search(line)

        if match:
            return match.group(1).strip()

    return ""


def get_vendor(mac_address: str) -> str:
    """
    MACアドレス先頭3オクテットから暫定メーカー判定を行う。
    """
    if not mac_address:
        return ""

    oui = mac_address[:8]

    return KNOWN_OUI_VENDORS.get(oui, "")


def is_locally_administered_mac(mac_address: str) -> bool:
    """
    ランダムMACまたはローカル管理MACの可能性を判定する。
    """
    if not mac_address:
        return False

    try:
        first_octet = int(mac_address.split(":")[0], 16)
    except ValueError:
        return False

    return bool(first_octet & 0x02)


def classify_device(
    dns_hostname: str,
    netbios_name: str,
    vendor: str,
    mac_address: str,
) -> tuple[str, str]:
    """
    現段階では端末を除外せず、分類候補と理由だけを付ける。
    """
    combined_name = (
        f"{dns_hostname} {netbios_name}"
        .strip()
        .upper()
    )

    if combined_name.startswith("LAPTOP-"):
        return "除外候補", "LAPTOP形式のホスト名"

    if "LAPTOP-" in combined_name:
        return "除外候補", "LAPTOP形式のホスト名"

    if "MACBOOK" in combined_name:
        return "除外候補", "MacBookらしいホスト名"

    personal_keywords = [
        "YUICHI",
        "MUTO",
        "KIMURA",
        "OKADA",
        "SHINDAI",
        "GIJUTSU",
    ]

    for keyword in personal_keywords:
        if keyword in combined_name:
            return (
                "除外候補",
                f"個人名または組織名候補: {keyword}",
            )

    upper_vendor = vendor.upper()

    hp_keywords = [
        "HP INC",
        "HEWLETT PACKARD",
        "HEWLETT-PACKARD",
    ]

    for keyword in hp_keywords:
        if keyword in upper_vendor:
            return "HP候補", f"HP系メーカー: {vendor}"

    excluded_vendors = [
        "BUFFALO",
        "APPLE",
        "DELL",
        "SIGMA KOKI",
        "RASPBERRY PI",
    ]

    for keyword in excluded_vendors:
        if keyword in upper_vendor:
            return (
                "除外候補",
                f"対象外メーカー候補: {vendor}",
            )

    if is_locally_administered_mac(mac_address):
        return (
            "要確認",
            "ランダムMACまたはローカル管理MACの可能性",
        )

    if "DESKTOP-" in combined_name:
        return (
            "要確認",
            "Windowsデスクトップ形式のホスト名",
        )

    if not combined_name and not vendor:
        return (
            "要確認",
            "ホスト名とメーカーを取得できない",
        )

    return (
        "要確認",
        "明確な分類条件に一致しない",
    )


def create_target_ip_list() -> list[str]:
    """
    設定されたネットワークからスキャン対象IP一覧を作る。
    """
    addresses: list[str] = []

    for network_text in SCAN_NETWORKS:
        network = ipaddress.ip_network(
            network_text,
            strict=False,
        )

        addresses.extend(
            str(ip_address)
            for ip_address in network.hosts()
        )

    return addresses


def scan_online_hosts(
    ip_addresses: list[str],
) -> list[str]:
    """
    IP一覧を並列Pingし、応答したIPだけを返す。
    """
    online_hosts: list[str] = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        future_to_ip = {
            executor.submit(
                ping_host,
                ip_address,
            ): ip_address
            for ip_address in ip_addresses
        }

        total = len(future_to_ip)
        completed = 0

        for future in as_completed(future_to_ip):
            ip_address = future_to_ip[future]
            completed += 1

            try:
                online = future.result()
            except Exception:
                online = False

            if online:
                online_hosts.append(ip_address)

                print(
                    f"{timestamp()} | ONLINE | {ip_address}",
                    flush=True,
                )

            if completed % 100 == 0:
                print(
                    f"{timestamp()} | "
                    f"進捗={completed}/{total} | "
                    f"オンライン={len(online_hosts)}",
                    flush=True,
                )

    online_hosts.sort(
        key=ipaddress.ip_address,
    )

    return online_hosts


def collect_device_information(
    online_hosts: list[str],
) -> list[dict[str, str]]:
    """
    オンライン端末についてホスト名、MAC、メーカーを収集する。
    """
    arp_table = read_arp_table()
    results: list[dict[str, str]] = []

    for index, ip_address in enumerate(
        online_hosts,
        start=1,
    ):
        print(
            f"{timestamp()} | "
            f"詳細取得={index}/{len(online_hosts)} | "
            f"{ip_address}",
            flush=True,
        )

        dns_hostname = get_dns_hostname(ip_address)
        netbios_name = get_netbios_name(ip_address)
        mac_address = arp_table.get(ip_address, "")
        vendor = get_vendor(mac_address)

        category, reason = classify_device(
            dns_hostname=dns_hostname,
            netbios_name=netbios_name,
            vendor=vendor,
            mac_address=mac_address,
        )

        results.append(
            {
                "IPアドレス": ip_address,
                "DNSホスト名": dns_hostname,
                "NetBIOS名": netbios_name,
                "MACアドレス": mac_address,
                "メーカー": vendor,
                "分類候補": category,
                "分類理由": reason,
                "確認時刻": timestamp(),
            }
        )

    return results


def save_csv(results: list[dict[str, str]]) -> None:
    """
    結果をExcelで開きやすいUTF-8 BOM付きCSVとして保存する。
    """
    fieldnames = [
        "IPアドレス",
        "DNSホスト名",
        "NetBIOS名",
        "MACアドレス",
        "メーカー",
        "分類候補",
        "分類理由",
        "確認時刻",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    ip_addresses = create_target_ip_list()

    print(
        f"{timestamp()} | スキャン開始 | "
        f"対象IP数={len(ip_addresses)}",
        flush=True,
    )

    online_hosts = scan_online_hosts(ip_addresses)

    print(
        f"{timestamp()} | Pingスキャン完了 | "
        f"オンライン={len(online_hosts)}",
        flush=True,
    )

    results = collect_device_information(online_hosts)

    save_csv(results)

    print()
    print("=" * 70)
    print(f"完了時刻: {timestamp()}")
    print(f"オンライン端末数: {len(results)}")
    print(f"CSV: {OUTPUT_CSV}")
    print("=" * 70)

    for result in results:
        device_name = (
            result["NetBIOS名"]
            or result["DNSホスト名"]
            or "-"
        )

        print(
            f"{result['IPアドレス']:<15} | "
            f"{device_name:<25} | "
            f"{result['MACアドレス'] or '-':<17} | "
            f"{result['メーカー'] or '-':<20} | "
            f"{result['分類候補']:<8} | "
            f"{result['分類理由']}"
        )


if __name__ == "__main__":
    main()