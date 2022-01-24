import urllib.robotparser
import requests
from bs4 import BeautifulSoup

url = 'http://www.minecraft.gamepedia.com'

itemUrl = 'https://minecraft.fandom.com/wiki/Block'
r = requests.get(itemUrl)
print(itemUrl)

robotUrl = 'https://minecraft.gamepedia.com/robots.txt'
print(robotUrl)
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'}

rb = requests.get(robotUrl, headers=headers)
test = rb.text

rp = urllib.robotparser.RobotFileParser()
rp.set_url(robotUrl)
rp.read()

fetchRobot = rp.can_fetch("*", robotUrl)
print(fetchRobot)

fetchItem = rp.can_fetch("*", itemUrl)
print(fetchItem)


def get_rp(robotUrl):
    global headers
    print('Robot URL: ' + robotUrl)
    print()
    r = requests.get(robotUrl, headers=headers)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(r.text)
    print('Allow fetch of robot.txt:', rp.can_fetch('', r.text))
    print()


def get_page(itemUrl):
    global headers
    print('Item page: ' + itemUrl)
    print()
    r = requests.get(itemUrl, headers=headers)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(r.text)
    if (rp.can_fetch('', r.text)) == True:
        html = requests.get(itemUrl)
        bs = BeautifulSoup(html.text, "html.parser")
        return bs
    else:
        return None


L = []
k = get_page('https://minecraft.fandom.com/wiki/Block')
data = k.find("div", attrs={'class': 'div-col columns column-width'})
data = data.find_all('a', attrs={'class': 'mw-redirect'})

for links in data:
    if "href" in links.attrs:
        if links.attrs["href"] not in L:
            next = links.attrs["href"]
            L.append(next)

L.remove('/wiki/Glow_Item_Frame')
L.remove('/wiki/Red_Mushroom')
L.remove('/wiki/Brown_Mushroom')
L.remove('/wiki/Cracked_Deepslate_Bricks')
L.remove('/wiki/Sweet_Berry_Bush')
L.remove('/wiki/Oak_Sapling')
L.remove('/wiki/Acacia_Sapling')
L.remove('/wiki/Spruce_Sapling')
L.remove('/wiki/Dark_Oak_Sapling')
L.remove('/wiki/Allium')
L.remove('/wiki/Dandelion')
L.remove('/wiki/Poppy')
L.remove('/wiki/Blue_Orchid')
L.remove('/wiki/Azure_Bluet')
L.remove('/wiki/White_Tulip')
L.remove('/wiki/Oxeye_Daisy')
L.remove('/wiki/Cornflower')
L.remove('/wiki/Lily_of_the_Valley')
L.remove('/wiki/Wither_Rose')
L.remove('/wiki/Sunflower')
L.remove('/wiki/Lilac')
L.remove('/wiki/Rose_Bush')
L.remove('/wiki/Peony')
L.remove('/wiki/Beetroots')

for i in range(len(L)):
    n = get_page(url + L[i])
    table = n.find("table", class_="wikitable")
    if len(table('tr')) < 5:
        table = table('tr')[3].text
    else:
        table = table('tr')[4].text

    print(table)
