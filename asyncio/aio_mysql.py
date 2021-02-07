#!/usr/bin/env python3

import datetime
import asyncio
import aiomysql
import functools

DB_HOST = '172.29.37.153'
DB_PORT = 3307
DB_USER = 'root'
DB_PWD = '123456'
DATABASE = 'dev'

def sync(func):
    @functools.wraps(func)
    def outer_decorator_func(*args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))

    return outer_decorator_func


class Mysql(object):
    def __init__(self):
        self.pool = None

    def connect(self):
        pool = aiomysql.create_pool(host=DB_HOST, port=DB_PORT,
                                    user=DB_USER, password=DB_PWD,
                                    db=DATABASE, charset='utf8mb4', autocommit=True)
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


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    my = Mysql()
    my.connect()
    sql_create_table = """CREATE TABLE IF NOT EXISTS `user_info` (
                `uid` BIGINT(20) UNSIGNED NOT NULL,
                `open_id` VARCHAR(80) BINARY NOT NULL,
                PRIMARY KEY (`uid`)
              )
              CHARSET=utf8mb4
              COLLATE='utf8mb4_unicode_ci'
              ENGINE=InnoDB;"""

    my.run_query(sql_create_table)

    res = my.run_query(f'SELECT * FROM user_info where uid<%s;', args=(10000,))
    print("------query-----", len(res), res)
    res = my.run_operation("INSERT INTO user_info(uid, open_id) VALUES(%s, %s) "
                           "ON DUPLICATE KEY UPDATE open_id=values(open_id); ", (9527, 'jiwekjlnik'))
    print("------insert into table-----", type(res), res)
    loop.run_forever()
