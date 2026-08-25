import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Slack Bot Token
SLACK_BOT_TOKEN = "xoxb-3622444354629-11192651743815-JrCzusEs5yjrhBELE8MGKnW1"

# Channel ID
CHANNEL_ID = "C0B60KMTR5K"

client = WebClient(token=SLACK_BOT_TOKEN)

while 1:
    try:
        response = client.chat_postMessage(
            channel=CHANNEL_ID,
            text="@Automation-bot 定期テスト(5s)：元気？"
        )

        print("送信成功")
        print(f"channel = {response['channel']}")
        print(f"ts = {response['ts']}")

    except SlackApiError as e:
        print("Slack API Error")
        print(e.response["error"])

    except Exception as e:
        print("その他エラー")
        print(e)

    time.sleep(5)