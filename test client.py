# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/3/23 
@time: 01:23
'''


import asyncio, json

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8000)

    print(f'Send: {message!r}')
    writer.write(message.encode())

    data = await reader.read(100)
    print(f'Received: {data.decode()!r}')

    print('Close the connection')
    writer.close()
    await writer.wait_closed()


async def main():
    tasks = [asyncio.create_task(tcp_echo_client(json.dumps({f'Client {client}':client}))) for client in range(10)]
    await asyncio.gather(*tasks)


asyncio.run(main())
