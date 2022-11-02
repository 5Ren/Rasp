import tkinter
import time
import threading


####
# ボタンクリック後の動作
def button_clicked():
    # ボタン非表示化
    button.pack_forget()

    # スレッディングでフリーズしないカウントダウン処理
    thread1 = threading.Thread(target=work1)
    thread1.start()


# スレッドでラベル表示を更新
def work1():
    global var

    # カウント
    for i in range(10):
        time.sleep(1)
        var.set(str(i + 1))

    var.set("待機中")

    # 最後にボタンを復元
    button.pack()


####
# Windowの設定
root = tkinter.Tk()
root.title("tkinterサンプル")
root.geometry("300x150")
# ラベル表示
var = tkinter.StringVar()
var.set("待機中")
label = tkinter.Label(root, textvariable=var)
label.pack()
# ボタン表示
button = tkinter.Button(root, text='処理開始', command=button_clicked)
button.pack()
root.mainloop()