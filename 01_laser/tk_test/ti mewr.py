#! python3.9.0
import PySimpleGUI as sg
import time
# 時間切れ時に手元のalarm音源mp3 or wmvを流す際は以下と節々のコメントアウト解除
import playsound

def time_as_int():
    return int(round(time.time() * 100))

# GUIテーマ。sg.theme_previewer()で使えるテーマを確認できる。
sg.theme('Reddit')

# tab1(表示部)
layout1 = [
    # 表示テキスト font設定は各々の環境を参照してください。
    [sg.Text('', size=(10,1), font=('Bauhaus 93',80), justification='c', pad=((30,30),(20,0)), key='text')],
    [sg.Text('', size=(28,1), font=('Helvetica',30), justification='c', pad=((0,0),(0,10)) ,key='add_text')],
    # 開始/停止・リセット・終了ボタン
    [sg.Button('Run', button_color=('white', 'Blue'), key='-RUN-PAUSE-'),
    sg.Button('Reset', button_color=('white', 'Green'), key='-RESET-'),
    sg.Exit(button_color=('white', 'Darkred'), key='-QUIT1-')]]

# tab2(設定部)
layout2 = [
    # 発表時間設定・質問時間設定。後者が要らなければそのままでおｋ
    # 発表部。デフォルトはカップ麺の待ち時間
    [sg.Text('*OPTIONAL*\nInput the time title:',size=(15,2), justification='c', pad=((0,0),(10,10))), sg.InputText(default_text='Presentation time' ,key='first_title')],
    [sg.Combo([i for i in range(0,60)] + [''], default_value='3',size=(5,7) , font=('Helvetica', 10), pad=((125,0),(0,0)), key='min'),
    sg.Text('min.', font=('Helvetica', 10)),
    sg.Combo([i for i in range(0,60)] + [''], default_value='0',size=(5,7), font=('Helvetica', 10), key='sec'),
    sg.Text('sec.', font=('Helvetica', 10))],
    [sg.Text('')],
    # 質問部。要らなくてもそのままでいい
    [sg.Text('*OPTIONAL*\nInput the additionnal time title:',size=(15,3) , justification='c', pad=((0,0),(0,0))), sg.InputText(default_text='Question time' ,key='second_title')],
    [sg.Combo([i for i in range(0,60)] + [''], default_value='0',size=(5,5) , font=('Helvetica', 10), pad=((125,0),(0,0)), key='add_min'),
    sg.Text('min.', font=('Helvetica', 10)),
    sg.Combo([i for i in range(0,60)] + [''], default_value='0',size=(5,5), font=('Helvetica', 10), key='add_sec'),
    sg.Text('sec.', font=('Helvetica', 10))],
    [sg.Exit(button_color=('white', 'Darkred'), key='-QUIT2-', pad=((0,0),(20,0)))]
    ]

#タブ化
layout = [[sg.TabGroup([[sg.Tab('Timer', layout1), sg.Tab('Settings', layout2)]])]]

# 表示。タイトル無し、ボタンリサイズ無し、最前面表示、どこ掴んでもいい
window = sg.Window('Running Timer', layout,
    no_titlebar=True,
    auto_size_buttons=False,
    keep_on_top=True,
    grab_anywhere=True,
    element_padding=(0, 0))

# 変数の初期化
start_time, paused_time, paused = 0, 0, True
first_flag = True
add_word = ''
hold_time, hold_flag = 0, True
alarm_flag = False

while True:
    if alarm_flag:
        playsound.playsound('鐘01.mp3') # ※音源のパスが必要※
        alarm_flag = not alarm_flag

    if not paused:      # Run状態 10msで読み込み、与えた時間とその変化を記述・追加時間がある場合の制御
        event, values = window.read(timeout=10)
        current_time = values['min'] * 100 * 60 + values['sec'] * 100
        add_time = values['add_min'] * 100 * 60 + values['add_sec'] * 100
        current_time += start_time - time_as_int()
        if current_time < 0: # 監視している時間変数が0を下回ると追加時間を加算
            if hold_flag:
                hold_time = current_time
                hold_flag = not hold_flag
            current_time += add_time
            add_word = values['second_title']
            if current_time > 0: # 追加時間があれば時間説明の書き換え
                if current_time - add_time == hold_time:
                    add_word = values['second_title']
                    current_time = 0
                    alarm_flag = not alarm_flag
            else: # 追加時間がなければ0とする
                event = '-RUN-PAUSE-'
                current_time = 0
                alarm_flag = not alarm_flag

    else:   # 待機（pause）状態 起動時の諸々を処理
        event, values = window.read(timeout=20)
        if first_flag:
            add_word = values['first_title']
            current_time = values['min'] * 100 * 60 + values['sec'] * 100
            window['text'].update('{:02d}:{:02d}'.format((current_time // 100) // 60, (current_time // 100) % 60))

        window['text'].update('{:02d}:{:02d}'.format((current_time // 100) // 60, (current_time // 100) % 60))
        window['add_text'].update(add_word)

        if event in (sg.WIN_CLOSED, '-QUIT1-'):    # QUIT押下
            break
        if event in (sg.WIN_CLOSED, '-QUIT2-'):
            break

        if event == '-RESET-':  # RESET押下
            paused_time = start_time = time_as_int()
            current_time = values['min'] * 100 * 60 + values['sec'] * 100
            add_word = values['first_title']
            alarm_flag = False

        elif event == '-RUN-PAUSE-':    # 0秒 or Run/Pause押下
            if first_flag: # 起動時の諸々を処理
                paused_time = time_as_int()
                window['-RUN-PAUSE-'].update('Pause')
                paused = not paused
                first_flag = not first_flag
                start_time = time_as_int()
                continue

        paused = not paused
        if paused:
            paused_time = time_as_int()
        else:
            start_time += time_as_int() - paused_time
        window['-RUN-PAUSE-'].update('Run' if paused else 'Pause')

window.close()