import math

stations = ['sibuya', 'omotesando', 'gaiennmae', 'aoyama1tyoume', 'akasaka']
syoyo_time = [0, 2, 4, 5, 7]

n = 0
k = 0

print('[銀座線]渋谷駅からの所要時間')
stations_in = input('input name: ')

while k == 0 and n < 5:
    if stations_in == stations[n]:
        k = 1
    n = n + 1

print(stations_in, '駅は')
if k == 0:
    print('銀座線の停車駅ではありません')


'''
box = [24, 15, 86, 7, 1, 33, 43, 62, 45, 22, 45, 78, 23, 14, 63, 67, 24, 14, 39]

#box[0] = 24
#box[1] = 15

for i in range(0, 10):
    print(box[i])
'''

'''
cnt = 0

def sosu_hantei(n, cnt):
    s = int(math.sqrt(n))
    k = 0
    if n == 1:
        k = 1
    elif n == 2:
        k = 0
    else:
        if n % 2 == 0:
            k = 1
        for i in range(3, s+1, 2):
            cnt = cnt + 1
            if n % i == 0:
                k = 1
    if k == 0:
        result = '素数です'
    else:
        result = '素数ではありません'
    return result, cnt

print(sosu_hantei(37, 0))
'''
