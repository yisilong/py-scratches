#!/usr/bin/env python3

import datetime
import asyncio
import aiomysql
import functools
from twisted.internet import asyncioreactor

asyncioreactor.install(asyncio.get_event_loop())


def sync(func):
    @functools.wraps(func)
    def outer_decorator_func(*args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))

    return outer_decorator_func


def peek(obj):
    for attr in dir(obj):
        if not attr.startswith("_"):
            print(attr, type(getattr(obj, attr)))
    print()


class Mysql(object):
    def __init__(self):
        self.pool = None

    def connect(self):
        pool = aiomysql.create_pool(host='127.0.0.1', port=3306,
                                    user='root', password='359359',
                                    db='dht', charset='utf8mb4', autocommit=True)
        self.pool = loop.run_until_complete(pool)

    @sync
    async def run_query(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, args)
                return await cur.fetchall()

    @sync
    async def run_operation(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                return await cur.execute(sql, args)

async def ttt():
    print('before----')
    await asyncio.sleep(2)
    print('after----')
    return 9527



if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    my = Mysql()
    my.connect()
    res = my.run_query("SELECT * FROM bt where id<%s;", args=(10,))
    print("------query-----", len(res), res)
    res = my.run_operation("INSERT INTO bt(info_hash, create_time, `name`, `length`, files) "
                           "VALUES(%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE hit=hit+1;",
                           ("abababab662ab", datetime.datetime.now(), "我的工作", 1024, '["file_1"]'))
    print("------insert into table-----", type(res), res)
    task = loop.create_task(ttt())
    task.add_done_callback(lambda t: print(1,2,5, "hell", t.result()))
    print("run----")
    print("fuck-", type(ttt()), type(asyncio.ensure_future(ttt())), type(loop.create_task(ttt())))
    loop.run_forever()
