#!/usr/bin/env python3

import aiohttp
import asyncio
import async_timeout
from urllib.parse import urljoin, urldefrag

root_url = "http://python.org/"
crawled_urls, url_hub = [], [root_url, f"{root_url}/sitemap.xml", f"{root_url}/robots.txt"]
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}


async def get_body(url):
    async with aiohttp.ClientSession() as session:
        try:
            with async_timeout.timeout(10):
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        return {'error': '', 'html': html}
                    else:
                        return {'error': response.status, 'html': ''}
        except Exception as err:
            return {'error': err, 'html': ''}


async def handle_task(task_id, work_queue):
    while not work_queue.empty():  # 如果队列不为空
        queue_url = await work_queue.get()  # 从队列中取出一个元素
        if not queue_url in crawled_urls:
            crawled_urls.append(queue_url)  # crawled_urls可以做一个去重操作
            body = await get_body(queue_url)
            if not body['error']:
                for new_url in get_urls(body['html']):
                    if root_url in new_url and not new_url in crawled_urls:
                        work_queue.put_nowait(new_url)
            else:
                print(f"Error: {body['error']} - {queue_url}")


def remove_fragment(url):
    pure_url, frag = urldefrag(url)
    return pure_url


def get_urls(html):
    new_urls = [url.split('"')[0] for url in str(html).replace("'", '"').split('href="')[1:]]
    return [urljoin(root_url, remove_fragment(new_url)) for new_url in new_urls]


if __name__ == "__main__":
    q = asyncio.Queue()  # 定义一个队列
    [q.put_nowait(url) for url in url_hub]  # 通过put_nowait方法循环往队列添加元素
    loop = asyncio.get_event_loop()
    tasks = [handle_task(task_id, q) for task_id in range(3)]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    for u in crawled_urls:
        print(u)
    print('-' * 30)
    print(len(crawled_urls))
