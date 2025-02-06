import tkinter as tk
import re

class Application(tk.Frame):
    # entry Widget用変数
    entry = None

    # 入力制限に失敗した場合に呼ばれる関数
    def invalidText(self):
        print('入力に失敗しました。')

    # 1. 入力制限の条件を設けて検証する関数の名前を決める
    # 4. 入力制限の条件を設けて検証する関数を実装する
    def onValidate(self, S):
        # 入力された文字が半角数字の場合
        # reについて : https://note.nkmk.me/python-re-match-search-findall-etc/
        if re.match(re.compile('[0-9]+'), S):
            return True
        else:
            # 入力不正のブザーを鳴らす。
            self.bell()
            return False

    def __init__(self, master=None):
        # Windowの初期設定を行う。
        super().__init__(master)

        # Windowの画面サイズを設定する。
        # geometryについて : https://kuroro.blog/python/rozH3S2CYE0a0nB3s2QL/
        self.master.geometry("300x200")

        # Windowを親要素として、frame Widget(Frame)を作成する。
        # Frameについて : https://kuroro.blog/python/P20XOidA5nh583fYRvxf/
        frame = tk.Frame(self.master)

        # Windowを親要素とした場合に、frame Widget(Frame)をどのように配置するのか?
        # packについて : https://kuroro.blog/python/UuvLfIBIEaw98BzBZ3FJ/
        frame.pack()

        # 2. 1で決めた関数名を、register関数を用いて登録する
        # register : 入力制限を行うための関数の登録を行う。パラメータと関数を紐づけるために必要。
        vcmd = self.register(self.onValidate)

        # frame Widget(Frame)を親要素として、entry Widgetを作成する。
        # width : 幅の設定
        # validate : 入力制限するオプションの値を設定。
        # validatecommand or vcmd : 入力制限用関数の設定。(3. entryのvalidatecommand option or vcmd optionへ、2の戻り値とパラメータを渡す)
        # invalidcommand : 入力制限により、入力不正が発生した場合に呼ばれる関数の設定。
        # Entryについて : https://kuroro.blog/python/PUZp77YFxrXvMCjpZbUg/
        self.entry = tk.Entry(frame, width=15, validate="key", validatecommand=(vcmd, '%S'), invalidcommand=self.invalidText)

        # frame Widget(Frame)を親要素とした場合に、entry Widgetをどのように配置するのか?
        # packについて : https://kuroro.blog/python/UuvLfIBIEaw98BzBZ3FJ/
        self.entry.pack()

# Tkinter初学者参考 : https://docs.python.org/ja/3/library/tkinter.html#a-simple-hello-world-program
if __name__ == "__main__":
    # Windowを生成する。
    # Windowについて : https://kuroro.blog/python/116yLvTkzH2AUJj8FHLx/
    root = tk.Tk()
    app = Application(master=root)
    # Windowをループさせて、継続的にWindow表示させる。
    # mainloopについて : https://kuroro.blog/python/DmJdUb50oAhmBteRa4fi/
    app.mainloop()