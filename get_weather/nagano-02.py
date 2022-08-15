import requests

# エリアコード
area_dic = {'長野県':'200000'}

def main():
    write_lists = []
    base_url = "https://www.jma.go.jp/bosai/forecast/data/forecast/"
    for k, v in area_dic.items():

        if k.find("/"):
            prefecture = k[0:k.find("/")]
        else:
            prefecture = k

        url = base_url + v + ".json"

        res = requests.get(url).json()

        for re in res:
            publishingOffice = re["publishingOffice"]
            reportDatetime = re["reportDatetime"]

            timeSeries = re["0"]

            for time in timeSeries:
                #降水確率など今回のターゲット以外は除外する
                if 'pops' in time["areas"][0]:
                    pass
                elif 'temps' in time["areas"][0]:
                    pass
                elif 'tempsMax' in time["areas"][0]:
                    pass
                else:
                    for i in range(len(time["areas"])):

                        local_name = time["areas"][i]["area"]["name"]

                        for j in range(len(timeSeries[0]["timeDefines"])):

                            if 'weathers' not in time["areas"][i]:
                                weather = ""
                            else:
                                weather = time["areas"][i]["weathers"][j]

                            if 'winds' not in time["areas"][i]:
                                wind = ""
                            else:
                                wind = time["areas"][i]["winds"][j]

                            if 'waves' not in time["areas"][i]:
                                wave = ""
                            else:
                                wave = time["areas"][i]["waves"][j]

                            timeDefine = time["timeDefines"][j]

                            # 各情報をリストに格納
                            write_list = []
                            write_list.append(prefecture)
                            write_list.append(publishingOffice)
                            write_list.append(reportDatetime)
                            write_list.append(local_name)
                            write_list.append(timeDefine)
                            write_list.append(weather)
                            write_list.append(wind)
                            write_list.append(wave)

                            write_lists.append(write_list)
    print(write_lists)

if __name__ == '__main__':
    main()
